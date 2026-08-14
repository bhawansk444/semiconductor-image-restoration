import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.network_swinir import SwinIR


# ============================================================
# PATHS
# ============================================================

DATA_DIR = r"E:\STUDIES\PROJECT\CORE DYNAMICS\SEMICON\UNZIP\TEST NOISE UNZIP\train"

NOISY_DIR = os.path.join(DATA_DIR, "NoisyLR")
GT_DIR = os.path.join(DATA_DIR, "GT")

START_MODEL = r"E:\STUDIES\python\semiconductor_ai\SwinIR-main\models\swinir_semiconductor_epoch25.pth"

SAVE_DIR = r"E:\STUDIES\python\semiconductor_ai\SwinIR-main\models"

SAVE_PATH = os.path.join(
    SAVE_DIR,
    "swinir_semiconductor_epoch50.pth"
)


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

START_EPOCH = 25
END_EPOCH = 50

BATCH_SIZE = 1

LR = 1e-5

CROP_SIZE = 64
SCALE = 2


# ============================================================
# DATASET
# ============================================================

class SemiconductorDataset(Dataset):

    def __init__(self, noisy_dir, gt_dir):

        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir

        noisy_files = set(
            f for f in os.listdir(noisy_dir)
            if f.endswith(".npy")
            and not f.startswith("._")
        )

        gt_files = set(
            f for f in os.listdir(gt_dir)
            if f.endswith(".npy")
            and not f.startswith("._")
        )

        self.files = sorted(
            list(noisy_files.intersection(gt_files))
        )

        print(
            "Valid matching image pairs:",
            len(self.files)
        )

        if len(self.files) == 0:
            raise RuntimeError(
                "No matching NPY image pairs found."
            )


    def __len__(self):

        return len(self.files)


    def __getitem__(self, index):

        filename = self.files[index]

        noisy_path = os.path.join(
            self.noisy_dir,
            filename
        )

        gt_path = os.path.join(
            self.gt_dir,
            filename
        )

        noisy = np.load(
            noisy_path
        ).astype(np.float32)

        gt = np.load(
            gt_path
        ).astype(np.float32)


        # ----------------------------------------------------
        # Make sure images are 2D
        # ----------------------------------------------------

        if noisy.ndim != 2:
            noisy = np.squeeze(noisy)

        if gt.ndim != 2:
            gt = np.squeeze(gt)


        # ----------------------------------------------------
        # Check image size
        # ----------------------------------------------------

        h, w = noisy.shape

        if h < CROP_SIZE or w < CROP_SIZE:

            raise RuntimeError(
                f"Noisy image {filename} is too small: "
                f"{noisy.shape}"
            )


        gt_h, gt_w = gt.shape

        required_size = CROP_SIZE * SCALE

        if (
            gt_h < required_size
            or
            gt_w < required_size
        ):

            raise RuntimeError(
                f"GT image {filename} is too small: "
                f"{gt.shape}"
            )


        # ----------------------------------------------------
        # Random crop
        # ----------------------------------------------------

        top = random.randint(
            0,
            h - CROP_SIZE
        )

        left = random.randint(
            0,
            w - CROP_SIZE
        )


        noisy_crop = noisy[
            top:top + CROP_SIZE,
            left:left + CROP_SIZE
        ]


        gt_top = top * SCALE
        gt_left = left * SCALE


        gt_crop = gt[
            gt_top:gt_top + required_size,
            gt_left:gt_left + required_size
        ]


        # ----------------------------------------------------
        # Normalize each pair
        # ----------------------------------------------------

        noisy_min = noisy_crop.min()
        noisy_max = noisy_crop.max()

        if noisy_max > noisy_min:

            noisy_crop = (
                noisy_crop - noisy_min
            ) / (
                noisy_max - noisy_min
            )

        noisy_crop = np.clip(
            noisy_crop,
            0,
            1
        )


        gt_min = gt_crop.min()
        gt_max = gt_crop.max()

        if gt_max > gt_min:

            gt_crop = (
                gt_crop - gt_min
            ) / (
                gt_max - gt_min
            )

        gt_crop = np.clip(
            gt_crop,
            0,
            1
        )


        # ----------------------------------------------------
        # NumPy → Torch
        # ----------------------------------------------------

        noisy_crop = torch.from_numpy(
            noisy_crop
        ).unsqueeze(0).float()


        gt_crop = torch.from_numpy(
            gt_crop
        ).unsqueeze(0).float()


        return noisy_crop, gt_crop


# ============================================================
# START
# ============================================================

print()
print("====================================")
print("CONTINUE TRAINING: EPOCH 26 -> 50")
print("====================================")


print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# DATASET
# ============================================================

dataset = SemiconductorDataset(
    NOISY_DIR,
    GT_DIR
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


print(
    "Training images:",
    len(dataset)
)

print(
    "Starting from epoch:",
    START_EPOCH
)

print(
    "Ending at epoch:",
    END_EPOCH
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Learning rate:",
    LR
)


# ============================================================
# CREATE SWINIR
# ============================================================

model = SwinIR(

    upscale=2,

    in_chans=1,

    img_size=64,

    window_size=8,

    img_range=1.0,

    depths=[
        6, 6, 6, 6, 6, 6
    ],

    embed_dim=180,

    num_heads=[
        6, 6, 6, 6, 6, 6
    ],

    mlp_ratio=2,

    upsampler="pixelshuffle",

    resi_connection="1conv"
)


model = model.to(DEVICE)


# ============================================================
# LOAD EPOCH 25 MODEL
# ============================================================

print()
print("Loading Epoch-25 model...")

checkpoint = torch.load(
    START_MODEL,
    map_location="cpu"
)


model.load_state_dict(
    checkpoint,
    strict=True
)


print(
    "Epoch-25 model loaded successfully!"
)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.L1Loss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)


# ============================================================
# TRAINING
# ============================================================

print()
print("====================================")
print("STARTING EPOCH 26 -> 50")
print("====================================")


model.train()


for epoch in range(
    START_EPOCH + 1,
    END_EPOCH + 1
):

    total_loss = 0.0


    for step, (noisy, gt) in enumerate(loader):

        noisy = noisy.to(DEVICE)

        gt = gt.to(DEVICE)


        optimizer.zero_grad()


        output = model(
            noisy
        )


        loss = criterion(
            output,
            gt
        )


        loss.backward()


        optimizer.step()


        total_loss += loss.item()


        print(
            f"Epoch [{epoch}/{END_EPOCH}] "
            f"Step [{step + 1}/{len(loader)}] "
            f"Loss: {loss.item():.6f}"
        )


    average_loss = (
        total_loss /
        len(loader)
    )


    print()
    print(
        f"Epoch {epoch} complete"
    )

    print(
        f"Average Loss: {average_loss:.6f}"
    )

    print()


# ============================================================
# SAVE EPOCH 50 MODEL
# ============================================================

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


torch.save(
    model.state_dict(),
    SAVE_PATH
)


print()
print("====================================")
print("TRAINING COMPLETE")
print("====================================")

print(
    "Final epoch:",
    END_EPOCH
)

print(
    "Model saved:"
)

print(
    SAVE_PATH
)

print("====================================")