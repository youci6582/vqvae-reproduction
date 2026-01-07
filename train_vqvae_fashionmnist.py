# train_vqvae_fashionmnist.py
# VQ-VAE on Fashion-MNIST (28x28 grayscale)

import os
import numpy as np
import torch

from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from vqvae import VQVAE

torch.set_printoptions(linewidth=160)


def save_img_tensors_as_grid(img_tensors: torch.Tensor, nrows: int, f: str) -> None:
    """
    Save a batch of images as a grid.

    img_tensors: [B, C, H, W], normalized roughly to [-0.5, 0.5]
    Supports grayscale (C=1) and RGB (C=3).
    """
    # [B, H, W, C]
    img_tensors = img_tensors.permute(0, 2, 3, 1).detach().cpu()

    imgs = img_tensors.numpy()
    imgs = np.clip(imgs, -0.5, 0.5)
    imgs = 255 * (imgs + 0.5)  # back to [0, 255]

    batch_size, img_h, img_w, channels = imgs.shape
    ncols = max(1, batch_size // nrows)

    # build grid (always save as RGB for convenience)
    grid = np.zeros((nrows * img_h, ncols * img_w, 3), dtype=np.uint8)

    for idx in range(min(batch_size, nrows * ncols)):
        row_idx = idx // ncols
        col_idx = idx % ncols
        r0, r1 = row_idx * img_h, (row_idx + 1) * img_h
        c0, c1 = col_idx * img_w, (col_idx + 1) * img_w

        if channels == 1:
            # grayscale -> replicate to RGB
            patch = imgs[idx, :, :, 0].astype(np.uint8)
            patch = np.stack([patch, patch, patch], axis=-1)
        else:
            patch = imgs[idx].astype(np.uint8)

        grid[r0:r1, c0:c1] = patch

    Image.fromarray(grid, "RGB").save(f"{f}.jpg")


def main():
    # -----------------------
    # Device
    # -----------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # -----------------------
    # Model (Fashion-MNIST is grayscale => in_channels=1)
    # -----------------------
    use_ema = True
    model_args = {
        "in_channels": 1,               # ⭐ important
        "num_hiddens": 128,
        "num_downsampling_layers": 2,
        "num_residual_layers": 2,
        "num_residual_hiddens": 32,
        "embedding_dim": 64,
        "num_embeddings": 512,
        "use_ema": use_ema,
        "decay": 0.99,
        "epsilon": 1e-5,
    }
    model = VQVAE(**model_args).to(device)

    # -----------------------
    # Dataset / DataLoader
    # -----------------------
    batch_size = 64
    workers = 4
    workers = min(workers, (os.cpu_count() or 4))

    # Fashion-MNIST: [0,1] -> normalize to roughly [-0.5, 0.5]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # [1, 28, 28]
            transforms.Normalize(mean=[0.5], std=[1.0]),
        ]
    )

    data_root = "../data"
    train_dataset = FashionMNIST(data_root, train=True, transform=transform, download=True)

    # For variance normalization: use raw pixel scale [0,1]
    # FashionMNIST stores data as uint8 [N, 28, 28]
    train_data_variance = np.var(train_dataset.data.numpy().astype(np.float32) / 255.0)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(workers > 0),
    )

    # -----------------------
    # Loss / Optimizer
    # -----------------------
    beta = 0.25
    lr = 3e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # -----------------------
    # Train 
    # -----------------------
    epochs = 10
    eval_every = 200
    best_train_loss = float("inf")

    model.train()
    for epoch in range(epochs):
        total_train_loss = 0.0
        total_recon_error = 0.0
        n_train = 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            imgs = imgs.to(device, non_blocking=True)
            out = model(imgs)

            recon_error = criterion(out["x_recon"], imgs) / train_data_variance
            loss = recon_error + beta * out["commitment_loss"]
            if not use_ema:
                loss = loss + out["dictionary_loss"]

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_recon_error += recon_error.item()
            n_train += 1

            if (batch_idx + 1) % eval_every == 0:
                avg_train_loss = total_train_loss / n_train
                best_train_loss = min(best_train_loss, avg_train_loss)

                print(f"epoch: {epoch} | batch_idx: {batch_idx + 1}", flush=True)
                print(f"avg_train_loss: {avg_train_loss}")
                print(f"best_train_loss: {best_train_loss}")
                print(f"avg_recon_error: {total_recon_error / n_train}\n")

                total_train_loss = 0.0
                total_recon_error = 0.0
                n_train = 0

    # -----------------------
    # Reconstruct & Save (random batch)
    # -----------------------
    model.eval()
    valid_dataset = FashionMNIST(data_root, train=False, transform=transform, download=True)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=True,  # 每次随机抽一批
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(workers > 0),
    )

    with torch.no_grad():
        imgs, _ = next(iter(valid_loader))
        save_img_tensors_as_grid(imgs, nrows=8, f="fashion_true")
        recon = model(imgs.to(device, non_blocking=True))["x_recon"]
        save_img_tensors_as_grid(recon, nrows=8, f="fashion_recon")


if __name__ == "__main__":
    # Windows / Anaconda 下更稳
    import torch.multiprocessing as mp

    mp.freeze_support()
    main()
