import os
import argparse
import numpy as np
import torch
from PIL import Image

from models.network_swinir import SwinIR


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="SwinIR Semiconductor Image Restoration"
)

parser.add_argument(
    "--input_dir",
    required=True,
    help="Path to input test images (.npy)"
)

parser.add_argument(
    "--output_dir",
    required=True,
    help="Path to save restored images"
)

parser.add_argument(
    "--model",
    default="models/swinir_semiconductor_FINAL.pth",
    help="Path to trained model weights"
)

args = parser.parse_args()


INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir
MODEL_PATH = args.model


# ============================================================
# CHECK PATHS
# ============================================================

if not os.path.isdir(INPUT_DIR):
    raise FileNotFoundError(
        f"Input directory not found: {INPUT_DIR}"
    )

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("====================================")
print("SWINIR SEMICONDUCTOR EVALUATION")
print("====================================")

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# CREATE SWINIR-M
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


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading model:")
print(MODEL_PATH)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

# Support both direct state_dict and
# checkpoint dictionaries
if "params" in checkpoint:
    checkpoint = checkpoint["params"]

if "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]


model.load_state_dict(
    checkpoint,
    strict=True
)

model = model.to(device)

model.eval()

print("Model loaded successfully!")


# ============================================================
# FIND INPUT FILES
# ============================================================

files = sorted(
    [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".npy")
        and not f.startswith("._")
    ]
)


if len(files) == 0:

    raise RuntimeError(
        "No .npy input images found."
    )


print()
print(
    "Input images:",
    len(files)
)

print(
    "Output directory:",
    OUTPUT_DIR
)


# ============================================================
# INFERENCE
# ============================================================

successful = 0

failed = 0


for index, filename in enumerate(files):

    print(
        f"[{index + 1}/{len(files)}] "
        f"Processing {filename}"
    )

    try:

        # ----------------------------------------------------
        # Load NPY
        # ----------------------------------------------------

        input_path = os.path.join(
            INPUT_DIR,
            filename
        )

        image = np.load(
            input_path
        ).astype(np.float32)


        # ----------------------------------------------------
        # Make sure image is 2D
        # ----------------------------------------------------

        image = np.squeeze(image)


        if image.ndim != 2:

            raise ValueError(
                f"Expected 2D grayscale image, "
                f"got shape {image.shape}"
            )


        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        image_min = image.min()

        image_max = image.max()


        if image_max > image_min:

            image = (
                image - image_min
            ) / (
                image_max - image_min
            )

        else:

            image = np.zeros_like(
                image,
                dtype=np.float32
            )


        image = np.clip(
            image,
            0.0,
            1.0
        )


        # ----------------------------------------------------
        # NumPy → Torch
        # ----------------------------------------------------

        tensor = torch.from_numpy(
            image
        ).unsqueeze(0).unsqueeze(0)


        tensor = tensor.to(
            device
        )


        # ----------------------------------------------------
        # SwinIR inference
        # ----------------------------------------------------

        with torch.no_grad():

            output = model(
                tensor
            )


        # ----------------------------------------------------
        # Torch → NumPy
        # ----------------------------------------------------

        output = (
            output
            .squeeze()
            .cpu()
            .numpy()
        )


        output = np.clip(
            output,
            0.0,
            1.0
        )


        # ----------------------------------------------------
        # Convert to PNG
        # ----------------------------------------------------

        output_uint8 = (
            output * 255.0
        ).round().astype(
            np.uint8
        )


        # ----------------------------------------------------
        # Save output
        # ----------------------------------------------------

        name = os.path.splitext(
            filename
        )[0]


        output_path = os.path.join(
            OUTPUT_DIR,
            name + "_restored.png"
        )


        Image.fromarray(
            output_uint8
        ).save(
            output_path
        )


        successful += 1


    except Exception as e:

        failed += 1

        print(
            "ERROR:",
            filename,
            "->",
            str(e)
        )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("====================================")
print("EVALUATION COMPLETE")
print("====================================")

print(
    "Total images:",
    len(files)
)

print(
    "Successfully processed:",
    successful
)

print(
    "Failed:",
    failed
)

print(
    "Results saved to:",
    OUTPUT_DIR
)

print("====================================")