import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MNIST_SVHN_Dataset(Dataset):
    def __init__(self, root="./data", train=True, download=True, max_samples=None):
        super().__init__()
        
        transform_mnist = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        
        transform_svhn = transforms.Compose([
            transforms.ToTensor(),
        ])

        mnist_ds = datasets.MNIST(root=root, train=train, download=download, transform=transform_mnist)
        svhn_split = 'train' if train else 'test'
        svhn_ds = datasets.SVHN(root=root, split=svhn_split, download=download, transform=transform_svhn)

        self.mnist_by_class = {i: [] for i in range(10)}
        for idx, (_, label) in enumerate(mnist_ds):
            self.mnist_by_class[label].append(idx)

        self.svhn_by_class = {i: [] for i in range(10)}
        for idx, (_, label) in enumerate(svhn_ds):
            self.svhn_by_class[label].append(idx)

        self.mnist_ds = mnist_ds
        self.svhn_ds = svhn_ds

        self.pairs = []
        for digit in range(10):
            m_indices = self.mnist_by_class[digit]
            s_indices = self.svhn_by_class[digit]
            n_pairs = min(len(m_indices), len(s_indices))
            
            np.random.shuffle(m_indices)
            np.random.shuffle(s_indices)
            for i in range(n_pairs):
                self.pairs.append((m_indices[i], s_indices[i], digit))
        if max_samples is not None and max_samples < len(self.pairs):
            np.random.shuffle(self.pairs)
            self.pairs = self.pairs[:max_samples]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        m_idx, s_idx, label = self.pairs[idx]
        img_mnist, _ = self.mnist_ds[m_idx]
        img_svhn, _ = self.svhn_ds[s_idx]
        return img_mnist, img_svhn, label


class MixtureOfExperts(nn.Module):
    def __init__(self, latent_dim, minimum_expert_weight=0.3):
        super().__init__()
        if not 0 <= minimum_expert_weight < 0.5:
            raise ValueError("minimum_expert_weight must be in [0, 0.5).")
        self.minimum_expert_weight = minimum_expert_weight
        self.gate = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, 1),
        )

    def forward(self, mu, logvar, return_weights=False):
        if mu.ndim != 3 or logvar.shape != mu.shape:
            raise ValueError(
                "mu and logvar must have matching shape "
                "(num_experts, batch_size, latent_dim)."
            )

        expert_features = torch.cat((mu, logvar), dim=-1)
        gate_probabilities = torch.softmax(
            self.gate(expert_features).squeeze(-1), dim=0
        )
        num_experts = gate_probabilities.size(0)
        weights = (
            self.minimum_expert_weight
            + (1 - num_experts * self.minimum_expert_weight)
            * gate_probabilities
        )

        component_variances = torch.exp(logvar)
        mixture_mu = torch.sum(weights.unsqueeze(-1) * mu, dim=0)
        mixture_var = torch.sum(
            weights.unsqueeze(-1) * (component_variances + mu.square()), dim=0
        ) - mixture_mu.square()
        mixture_logvar = torch.log(mixture_var.clamp_min(1e-8))
        if return_weights:
            return mixture_mu, mixture_logvar, weights
        return mixture_mu, mixture_logvar


class EncoderMNIST(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class EncoderSVHN(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class DecoderMNIST(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 64 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(
            16, 1, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(16)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = h.view(-1, 64, 4, 4)
        h = F.relu(self.bn1(self.deconv1(h)))
        h = F.relu(self.bn2(self.deconv2(h)))
        return torch.sigmoid(self.deconv3(h))


class DecoderSVHN(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 64 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(
            16, 3, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(16)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = h.view(-1, 64, 4, 4)
        h = F.relu(self.bn1(self.deconv1(h)))
        h = F.relu(self.bn2(self.deconv2(h)))
        return torch.sigmoid(self.deconv3(h))


class MultiViewVAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.encoder_mnist = EncoderMNIST(latent_dim)
        self.encoder_svhn = EncoderSVHN(latent_dim)
        self.decoder_mnist = DecoderMNIST(latent_dim)
        self.decoder_svhn = DecoderSVHN(latent_dim)
        self.moe = MixtureOfExperts(latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def infer_joint(self, x_mnist=None, x_svhn=None):
        if x_mnist is None and x_svhn is None:
            raise ValueError("At least one modality must be provided.")

        mus, logvars = [], []
        if x_mnist is not None:
            m_m, l_m = self.encoder_mnist(x_mnist)
            mus.append(m_m)
            logvars.append(l_m)
        if x_svhn is not None:
            m_s, l_s = self.encoder_svhn(x_svhn)
            mus.append(m_s)
            logvars.append(l_s)

        mus = torch.stack(mus, dim=0)
        logvars = torch.stack(logvars, dim=0)
        return self.moe(mus, logvars)

    def forward(self, x_mnist, x_svhn):
        mu_joint, logvar_joint = self.infer_joint(x_mnist, x_svhn)
        z = self.reparameterize(mu_joint, logvar_joint)
        
        recon_mnist = self.decoder_mnist(z)
        recon_svhn = self.decoder_svhn(z)
        
        return recon_mnist, recon_svhn, mu_joint, logvar_joint


def compute_kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)


def eval_Gaussian_LL(x, mu_x, var_x):
    dimensions = x.shape[1] * x.shape[2] * x.shape[3]
    x = x.reshape(-1, dimensions)
    mu_x = mu_x.reshape(-1, dimensions)
    variance = torch.ones_like(mu_x) * var_x

    constant = dimensions * np.log(2 * np.pi) + torch.sum(
        torch.log(variance), dim=-1
    )
    logp_data = -0.5 * (
        constant
        + torch.sum((x - mu_x).square() / variance, dim=-1)
    )
    return torch.sum(logp_data), logp_data


def mvae_loss_function(
    model,
    x_mnist,
    x_svhn,
    beta=1.0,
    svhn_weight=2.0,
    moe_balance_weight=50.0,
    return_metrics=False,
):
    batch_size = x_mnist.size(0)

    mu_m, logvar_m = model.encoder_mnist(x_mnist)
    mu_s, logvar_s = model.encoder_svhn(x_svhn)
    mu_j, logvar_j, gate_weights = model.moe(
        torch.stack((mu_m, mu_s)),
        torch.stack((logvar_m, logvar_s)),
        return_weights=True,
    )
    gate_usage = gate_weights.mean(dim=1)
    target_usage = 1.0 / gate_weights.size(0)
    balance_loss = gate_weights.size(0) * torch.sum(
        (gate_usage - target_usage).square()
    )
    z_j = model.reparameterize(mu_j, logvar_j)
    rec_m_j = model.decoder_mnist(z_j)
    rec_s_j = model.decoder_svhn(z_j)
    loss_rec_m_j = F.binary_cross_entropy(
        rec_m_j, x_mnist, reduction="sum"
    ) / batch_size
    loss_rec_s_j_unweighted = (
        F.mse_loss(rec_s_j, x_svhn, reduction="sum") / batch_size
    )
    loss_rec_s_j = loss_rec_s_j_unweighted * svhn_weight
    kl_j = compute_kl_divergence(mu_j, logvar_j).sum() / batch_size

    z_m = model.reparameterize(mu_m, logvar_m)
    rec_m_m = model.decoder_mnist(z_m)
    loss_rec_m_m = F.binary_cross_entropy(
        rec_m_m, x_mnist, reduction="sum"
    ) / batch_size
    kl_m = compute_kl_divergence(mu_m, logvar_m).sum() / batch_size

    z_s = model.reparameterize(mu_s, logvar_s)
    rec_s_s = model.decoder_svhn(z_s)
    loss_rec_s_s_unweighted = (
        F.mse_loss(rec_s_s, x_svhn, reduction="sum") / batch_size
    )
    loss_rec_s_s = loss_rec_s_s_unweighted * svhn_weight
    kl_s = compute_kl_divergence(mu_s, logvar_s).sum() / batch_size

    total_recon = loss_rec_m_j + loss_rec_s_j + loss_rec_m_m + loss_rec_s_s
    total_kl = kl_j + kl_m + kl_s

    total_loss = (
        total_recon
        + beta * total_kl
        + moe_balance_weight * balance_loss
    )
    if return_metrics:
        return (
            total_loss,
            total_recon.item(),
            total_kl.item(),
            gate_usage.detach(),
            balance_loss.item(),
            loss_rec_m_j.item(),
            loss_rec_m_m.item(),
            loss_rec_s_j_unweighted.item(),
            loss_rec_s_s_unweighted.item(),
        )
    return total_loss, total_recon.item(), total_kl.item()


def train_mvae(
    model,
    dataloader,
    optimizer,
    epoch,
    device,
    beta=1.0,
    svhn_weight=2.0,
    moe_balance_weight=50.0,
):
    model.train()
    running_loss = 0.0
    running_recon = 0.0
    running_kl = 0.0
    running_balance = 0.0
    running_mnist_joint = 0.0
    running_mnist_unimodal = 0.0
    running_svhn_joint = 0.0
    running_svhn_unimodal = 0.0
    running_gate_usage = None
    sample_count = 0
    
    for batch_idx, (x_m, x_s, _) in enumerate(dataloader):
        x_m, x_s = x_m.to(device), x_s.to(device)
        optimizer.zero_grad()
        
        (
            loss,
            recon_val,
            kl_val,
            gate_usage,
            balance_val,
            mnist_joint_val,
            mnist_unimodal_val,
            svhn_joint_val,
            svhn_unimodal_val,
        ) = mvae_loss_function(
            model,
            x_m,
            x_s,
            beta=beta,
            svhn_weight=svhn_weight,
            moe_balance_weight=moe_balance_weight,
            return_metrics=True,
        )
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_recon += recon_val
        running_kl += kl_val
        running_balance += balance_val
        running_mnist_joint += mnist_joint_val
        running_mnist_unimodal += mnist_unimodal_val
        running_svhn_joint += svhn_joint_val
        running_svhn_unimodal += svhn_unimodal_val
        if running_gate_usage is None:
            running_gate_usage = torch.zeros_like(gate_usage)
        running_gate_usage += gate_usage * x_m.size(0)
        sample_count += x_m.size(0)
        
    num_batches = len(dataloader)
    if num_batches == 0:
        raise ValueError("dataloader must contain at least one batch.")
    avg_loss = running_loss / num_batches
    avg_recon = running_recon / num_batches
    avg_kl = running_kl / num_batches
    avg_balance = running_balance / num_batches
    avg_mnist_joint = running_mnist_joint / num_batches
    avg_mnist_unimodal = running_mnist_unimodal / num_batches
    avg_svhn_joint = running_svhn_joint / num_batches
    avg_svhn_unimodal = running_svhn_unimodal / num_batches
    avg_gate_usage = running_gate_usage / sample_count
    
    print(
        f"Epoch [{epoch:02d}] | Beta: {beta:.2f} | "
        f"Loss: {avg_loss:.2f} (Recon: {avg_recon:.2f}, "
        f"KL: {avg_kl:.2f}, MoE balance: {avg_balance:.3f}) | "
        f"Joint BCE/MSE M/S: {avg_mnist_joint:.2f}/{avg_svhn_joint:.2f}; "
        f"unimodal BCE/MSE M/S: "
        f"{avg_mnist_unimodal:.2f}/{avg_svhn_unimodal:.2f} "
        f"(SVHN weighted total: "
        f"{(avg_svhn_joint + avg_svhn_unimodal) * svhn_weight:.2f}) | "
        f"Gate usage MNIST/SVHN: "
        f"{avg_gate_usage[0].item():.3f}/{avg_gate_usage[1].item():.3f}"
    )
    return avg_loss

def get_samples_per_class(dataloader, device, num_classes=8):
    samples_m = {}
    samples_s = {}
    
    for x_m, x_s, labels in dataloader:
        for i in range(len(labels)):
            lbl = labels[i].item()
            if lbl not in samples_m:
                samples_m[lbl] = x_m[i]
                samples_s[lbl] = x_s[i]
            if len(samples_m) == num_classes:
                break
        if len(samples_m) == num_classes:
            break
            
    sorted_keys = sorted(samples_m.keys())
    batch_m = torch.stack([samples_m[k] for k in sorted_keys]).to(device)
    batch_s = torch.stack([samples_s[k] for k in sorted_keys]).to(device)
    batch_labels = torch.tensor(sorted_keys).to(device)
    
    return batch_m, batch_s, batch_labels


import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE

def _single_process_loader(dataloader):
    if not isinstance(dataloader, DataLoader) or dataloader.num_workers == 0:
        return dataloader

    loader_args = {
        "dataset": dataloader.dataset,
        "num_workers": 0,
        "collate_fn": dataloader.collate_fn,
        "pin_memory": False,
    }
    if dataloader.batch_size is None:
        return DataLoader(batch_size=None, **loader_args)

    return DataLoader(
        batch_size=dataloader.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_args,
    )


def _contrast_preview_rgb(image, lower_percentile=1, upper_percentile=99):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must have shape (height, width, 3).")

    image = np.clip(image, 0, 1)
    luminance = (
        0.2126 * image[..., 0]
        + 0.7152 * image[..., 1]
        + 0.0722 * image[..., 2]
    )
    lower, upper = np.percentile(
        luminance, (lower_percentile, upper_percentile)
    )
    if upper - lower <= 1e-6:
        return image, luminance

    preview = np.clip(
        (image - lower) / (upper - lower), 0, 1
    )
    luminance_preview = np.clip(
        (luminance - lower) / (upper - lower), 0, 1
    )
    return preview, luminance_preview


def _contrast_stretch_grayscale(image, lower_percentile=1, upper_percentile=99):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("image must have shape (height, width).")

    lower, upper = np.percentile(image, (lower_percentile, upper_percentile))
    if upper - lower <= 1e-6:
        return np.clip(image, 0, 1)
    return np.clip((image - lower) / (upper - lower), 0, 1)


@torch.no_grad()
def generate_from_prior(model, num_samples=8, device=None, temperature=1.0):
    if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples < 1:
        raise ValueError("num_samples must be a positive integer.")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite positive number.")

    model_device = next(model.parameters()).device
    if device is not None:
        target_device = torch.device(device)
        if target_device.type != model_device.type:
            raise ValueError(f"device must match the model device ({model_device}).")

    was_training = model.training
    model.eval()
    try:
        z = torch.randn(num_samples, model.latent_dim, device=model_device) * temperature
        generated_mnist = model.decoder_mnist(z)
        generated_svhn = model.decoder_svhn(z)
    finally:
        model.train(was_training)

    return generated_mnist, generated_svhn


@torch.no_grad()
def visualize_results(
    model,
    dataloader,
    device,
    num_samples=8,
    tsne_samples=3000,
    temperature=1.0,
    prior_samples=8,
    save_dir=None,
    show=True,
):
    model.eval()
    figures = {}
    output_dir = Path(save_dir) if save_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    visualization_loader = _single_process_loader(dataloader)
    x_m, x_s, labels = get_samples_per_class(
        visualization_loader, device, num_classes=num_samples
    )
    n = len(labels)

    rec_m, rec_s, _, _ = model(x_m, x_s)

    fig, axes = plt.subplots(
        4, n, figsize=(2.0 * n, 8), squeeze=False,
        gridspec_kw={"wspace": 0, "hspace": 0},
    )
    reconstruction_rows = (
        "MNIST input",
        "MNIST reconstruction",
        "SVHN input",
        "SVHN reconstruction",
    )

    for i in range(n):
        axes[0, i].imshow(
            x_m[i].cpu().squeeze(),
            cmap="gray",
        )
        axes[1, i].imshow(
            rec_m[i].cpu().squeeze(),
            cmap="gray",
        )
        axes[2, i].imshow(
            x_s[i].cpu().permute(1, 2, 0),
        )
        svhn_rec = rec_s[i].cpu().permute(1, 2, 0).numpy()
        axes[3, i].imshow(svhn_rec)
        for row, axis in enumerate(axes[:, i]):
            axis.set_axis_off()
            if i == 0:
                axis.text(
                    0.02,
                    0.98,
                    reconstruction_rows[row],
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
                )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    figures["reconstructions"] = fig

    generated_mnist, generated_svhn = generate_from_prior(
        model,
        num_samples=prior_samples,
        device=device,
        temperature=temperature,
    )

    fig, axes = plt.subplots(
        3,
        prior_samples,
        figsize=(2.0 * prior_samples, 6),
        squeeze=False,
        gridspec_kw={"wspace": 0, "hspace": 0},
    )
    prior_rows = ("MNIST", "SVHN raw", "SVHN contrast")

    for i in range(prior_samples):
        axes[0, i].imshow(
            generated_mnist[i].cpu().squeeze(),
            cmap="gray",
        )

        svhn_gen = generated_svhn[i].cpu().permute(1, 2, 0).numpy()
        svhn_preview, _ = _contrast_preview_rgb(svhn_gen)
        axes[1, i].imshow(svhn_gen)
        axes[2, i].imshow(svhn_preview)
        for row, axis in enumerate(axes[:, i]):
            axis.set_axis_off()
            if i == 0:
                axis.text(
                    0.02,
                    0.98,
                    prior_rows[row],
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
                )
        axes[1, i].text(
            0.98,
            0.02,
            f"std {generated_svhn[i].std().item():.3f}",
            transform=axes[1, i].transAxes,
            ha="right",
            va="bottom",
            fontsize=6,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
        )

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    figures["prior_samples"] = fig
    mu_m, _ = model.infer_joint(x_mnist=x_m)
    cross_svhn = model.decoder_svhn(mu_m)

    mu_s, _ = model.infer_joint(x_svhn=x_s)
    cross_mnist = model.decoder_mnist(mu_s)

    fig, axes = plt.subplots(
        5, n, figsize=(2.0 * n, 10), squeeze=False,
        gridspec_kw={"wspace": 0, "hspace": 0},
    )
    cross_domain_rows = (
        "MNIST input",
        "MNIST → SVHN raw",
        "MNIST → SVHN contrast",
        "SVHN input",
        "SVHN → MNIST",
    )

    for i in range(n):
        axes[0, i].imshow(
            x_m[i].cpu().squeeze(),
            cmap="gray"
        )

        svhn_trans = cross_svhn[i].cpu().permute(1, 2, 0).clamp(0,1)
        svhn_preview, _ = _contrast_preview_rgb(svhn_trans.numpy())
        axes[1, i].imshow(svhn_trans)
        axes[2, i].imshow(svhn_preview)

        svhn_in = x_s[i].cpu().permute(1, 2, 0)
        axes[3, i].imshow(svhn_in)

        cross_mnist_image = cross_mnist[i].cpu().squeeze()
        axes[4, i].imshow(
            cross_mnist_image,
            cmap="gray"
        )
        for row, axis in enumerate(axes[:, i]):
            axis.set_axis_off()
            if i == 0:
                axis.text(
                    0.02,
                    0.98,
                    cross_domain_rows[row],
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
                )
        axes[0, i].text(
            0.98,
            0.02,
            f"Digit {labels[i].item()}",
            transform=axes[0, i].transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
        )

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    figures["cross_domain"] = fig

    latents = []
    labels_all = []

    for x_m_b, x_s_b, y_b in visualization_loader:
        x_m_b = x_m_b.to(device)
        x_s_b = x_s_b.to(device)

        mu, _ = model.infer_joint(x_m_b, x_s_b)

        latents.append(mu.cpu())
        labels_all.append(y_b)

    latents = torch.cat(latents, dim=0).numpy()
    labels_all = torch.cat(labels_all, dim=0).numpy()

    sample_count = min(tsne_samples, len(labels_all))
    if sample_count < 2:
        raise ValueError("t-SNE requires at least two samples.")

    classes, class_counts = np.unique(labels_all, return_counts=True)
    rng = np.random.default_rng(42)
    selected_indices = []
    remaining = sample_count
    available_counts = class_counts.copy()

    while remaining > 0:
        active_classes = np.flatnonzero(available_counts > 0)
        per_class, extra = divmod(remaining, len(active_classes))
        quotas = np.full(len(classes), per_class, dtype=int)
        quotas[active_classes[:extra]] += 1
        quotas = np.minimum(quotas, available_counts)

        for class_index, (label, quota) in enumerate(zip(classes, quotas)):
            if quota:
                class_indices = np.flatnonzero(labels_all == label)
                selected_indices.extend(
                    rng.choice(class_indices, size=quota, replace=False)
                )
                available_counts[class_index] -= quota
                remaining -= quota

    selected_indices = np.asarray(selected_indices)
    labels_all = labels_all[selected_indices]

    tsne = TSNE(
        n_components=2,
        perplexity=40,
        max_iter=1000,
        random_state=42
    )

    z_2d = tsne.fit_transform(latents[selected_indices])

    fig, axis = plt.subplots(figsize=(7, 5.5))
    scatter = axis.scatter(
        z_2d[:, 0],
        z_2d[:, 1],
        c=labels_all,
        cmap="tab10",
        vmin=-0.5,
        vmax=9.5,
        s=10,
        alpha=0.7
    )

    axis.set_title("Joint Latent Space — t-SNE", fontsize=13)
    axis.set_xlabel("t-SNE Component 1")
    axis.set_ylabel("t-SNE Component 2")
    colorbar = fig.colorbar(scatter, ax=axis, ticks=np.arange(10))
    colorbar.set_label("Digit")
    fig.tight_layout()
    figures["tsne"] = fig

    if output_dir is not None:
        for name, figure in figures.items():
            figure.savefig(
                output_dir / f"{name}.png",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
            )
            figure.savefig(
                output_dir / f"{name}.pdf",
                bbox_inches="tight",
                pad_inches=0,
            )

    if show:
        plt.show()

    return figures