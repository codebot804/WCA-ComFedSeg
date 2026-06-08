"""Create Phase 5A real dataset sample visualizations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.busi import BUSISegmentationDataset, discover_busi_samples
from datasets.kvasir_seg import KvasirSegmentationDataset, discover_kvasir_samples

OUTPUT_DIR = PROJECT_ROOT / "results/figures/phase5a_real_data_samples"


def _tensor_to_gray(array: np.ndarray, panel_size: int = 180) -> Image.Image:
    array = np.squeeze(array)
    array = np.clip(array, 0.0, 1.0)
    image = Image.fromarray((array * 255).astype(np.uint8), mode="L")
    return image.resize((panel_size, panel_size), resample=Image.Resampling.NEAREST).convert("RGB")


def _overlay(image: np.ndarray, mask: np.ndarray, panel_size: int = 180) -> Image.Image:
    base = _tensor_to_gray(image, panel_size=panel_size).convert("RGBA")
    mask_np = np.squeeze(mask) > 0
    mask_img = Image.fromarray((mask_np.astype(np.uint8) * 150), mode="L")
    mask_img = mask_img.resize((panel_size, panel_size), resample=Image.Resampling.NEAREST)
    red = Image.new("RGBA", (panel_size, panel_size), color=(220, 30, 30, 0))
    red.putalpha(mask_img)
    return Image.alpha_composite(base, red).convert("RGB")


def _make_figure(dataset, sample_indices: list[int], output_path: Path, title: str) -> None:
    panel_size = 180
    title_height = 28
    row_label_width = 120
    padding = 10
    columns = ["Original image", "Binary mask", "Overlay"]
    width = row_label_width + 3 * panel_size + 5 * padding
    height = title_height + len(sample_indices) * (panel_size + padding) + 2 * padding
    canvas = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(canvas)
    draw.text((padding, padding), title, fill="black")
    for col, label in enumerate(columns):
        x = row_label_width + padding + col * (panel_size + padding)
        draw.text((x, padding), label, fill="black")

    for row, sample_index in enumerate(sample_indices):
        image_tensor, mask_tensor = dataset[sample_index]
        image_np = image_tensor.numpy()
        mask_np = mask_tensor.numpy()
        y = title_height + padding + row * (panel_size + padding)
        draw.text((padding, y + 6), f"Sample {sample_index}", fill="black")
        panels = [
            _tensor_to_gray(image_np, panel_size),
            _tensor_to_gray(mask_np, panel_size),
            _overlay(image_np, mask_np, panel_size),
        ]
        for col, panel in enumerate(panels):
            x = row_label_width + padding + col * (panel_size + padding)
            canvas.paste(panel, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _sample_indices(length: int, count: int = 4) -> list[int]:
    if length <= 0:
        return []
    if length <= count:
        return list(range(length))
    return [int(round(value)) for value in np.linspace(0, length - 1, num=count)]


def main() -> None:
    busi_samples, _ = discover_busi_samples()
    kvasir_samples, _ = discover_kvasir_samples()
    if not busi_samples:
        raise RuntimeError("No BUSI samples available for visualization.")
    if not kvasir_samples:
        raise RuntimeError("No Kvasir-SEG samples available for visualization.")

    busi_dataset = BUSISegmentationDataset(busi_samples, image_size=128)
    kvasir_dataset = KvasirSegmentationDataset(kvasir_samples, image_size=128)
    busi_path = OUTPUT_DIR / "busi_samples.png"
    kvasir_path = OUTPUT_DIR / "kvasir_seg_samples.png"

    _make_figure(busi_dataset, _sample_indices(len(busi_dataset)), busi_path, "BUSI Phase 5A samples")
    _make_figure(kvasir_dataset, _sample_indices(len(kvasir_dataset)), kvasir_path, "Kvasir-SEG Phase 5A samples")

    print(f"BUSI sample visualization saved to: {busi_path}")
    print(f"Kvasir-SEG sample visualization saved to: {kvasir_path}")


if __name__ == "__main__":
    main()
