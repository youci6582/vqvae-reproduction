# See: https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb.

import os
import numpy as np
import torch

from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from vqvae import VQVAE

torch.set_printoptions(linewidth=160)


def save_img_tensors_as_grid(img_tensors: torch.Tensor, nrows: int, f: str) -> None:
    """
    Save a batch of images as a grid.

    img_tensors: [B, C, H, W], assumed to be roughly in [-0.5, 0.5] after normalization.
    nrows: number of rows in the grid.
    f: output filename prefix (without extension).
    """
    img_tensors = img_tensors.permute(0, 2, 3, 1)  # [B, H, W, C]
    imgs_array = img_tensors.detach().cpu().numpy()
    imgs_array = np.clip(imgs_array, -0.5, 0.5)
    imgs_array = 255 * (imgs_array + 0.5)

    batch_size, img_h, img_w = img_tensors.shape[:3]
    ncols = max(1, batch_size // nrows)

    grid = np.zeros((nrows * img_h, ncols * img_w, 3), dtype=np.uint8)

    for idx in range(min(batch_size, nrows * ncols)):
        row_idx = idx // ncols
        col_idx = idx % ncols
        row_start = row_idx * img_h
        row_end = row_start + img_h
        col_start = col_idx * img_w
        col_end = col_start + img_w
        grid[row_start:row_end, col_start:col_end] = imgs_array[idx].astype(np.uint8)

    Image.fromarray(grid, "RGB").save(f"{f}.jpg")


def main():
    # -----------------------
    # Device
    # -----------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # -----------------------
    # Model
    # -----------------------
    use_ema = True
    model_args = {
        "in_channels": 3,
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
    batch_size = 32

    workers = 8
    workers = min(workers, (os.cpu_count() or 4))

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    data_root = "../data"
    train_dataset = CIFAR10(data_root, train=True, transform=transform, download=True)
    train_data_variance = np.var(train_dataset.data.astype(np.float32) / 255.0)

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
    beta = 0.25  # commitment loss weight
    lr = 3e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # -----------------------
    # Train
    # -----------------------
    epochs = 7
    eval_every = 100
    best_train_loss = float("inf")

    model.train()
    for epoch in range(epochs):
        total_train_loss = 0.0
        total_recon_error = 0.0
        n_train = 0

        for batch_idx, train_tensors in enumerate(train_loader):
            optimizer.zero_grad()

            imgs = train_tensors[0].to(device, non_blocking=True)
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
    # Reconstruct & Save
    # -----------------------
    model.eval()
    valid_dataset = CIFAR10(data_root, train=False, transform=transform, download=True)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=True,  # 每次取不同图片，更符合你的“换图”需求
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(workers > 0),
    )

    with torch.no_grad():
        valid_tensors = next(iter(valid_loader))
        save_img_tensors_as_grid(valid_tensors[0], 4, "true")
        recon = model(valid_tensors[0].to(device, non_blocking=True))["x_recon"]
        save_img_tensors_as_grid(recon, 4, "recon")


if __name__ == "__main__":
    # Windows / Anaconda 下多进程更稳（保留这个是“必要因素”之一）
    import torch.multiprocessing as mp

    mp.freeze_support()
    main()
