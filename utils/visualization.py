"""Prediction and split-distribution visualization helpers.

These utilities create qualitative prediction panels and client split summaries
for experiment inspection. Publication figures are generated separately from
processed result summaries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image, ImageDraw

from federated.client import FederatedClient
from utils.metrics import binary_predictions


def _to_grayscale_image(array: np.ndarray, size: int = 192) -> Image.Image:
    array = np.clip(array, 0.0, 1.0)
    image = Image.fromarray((array * 255).astype(np.uint8), mode="L")
    return image.resize((size, size), resample=Image.Resampling.NEAREST).convert("RGB")


def _make_tripanel(image_np: np.ndarray, mask_np: np.ndarray, pred_np: np.ndarray) -> Image.Image:
    panel_size = 192
    title_height = 26
    padding = 8
    titles = ["Input image", "Ground truth mask", "Predicted mask"]
    arrays = [image_np, mask_np, pred_np]

    width = panel_size * 3 + padding * 4
    height = panel_size + title_height + padding * 2
    canvas = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(canvas)

    for index, (title, array) in enumerate(zip(titles, arrays)):
        x = padding + index * (panel_size + padding)
        draw.text((x, padding), title, fill="black")
        canvas.paste(_to_grayscale_image(array, size=panel_size), (x, padding + title_height))

    return canvas


def _bar_color(index: int) -> tuple[int, int, int]:
    colors = [
        (58, 113, 193),
        (61, 150, 96),
        (202, 122, 58),
        (142, 95, 178),
        (190, 80, 80),
    ]
    return colors[index % len(colors)]


def _draw_bar_group(
    draw: ImageDraw.ImageDraw,
    origin_x: int,
    origin_y: int,
    width: int,
    height: int,
    title: str,
    values: list[float],
    labels: list[str],
) -> None:
    draw.text((origin_x, origin_y), title, fill="black")
    chart_top = origin_y + 24
    chart_height = height - 48
    chart_bottom = chart_top + chart_height
    draw.line((origin_x, chart_bottom, origin_x + width, chart_bottom), fill="black", width=1)

    max_value = max(values) if values else 1.0
    max_value = max(max_value, 1e-7)
    bar_gap = 12
    bar_width = max(16, int((width - bar_gap * (len(values) + 1)) / max(len(values), 1)))

    for index, value in enumerate(values):
        x0 = origin_x + bar_gap + index * (bar_width + bar_gap)
        bar_height = int((value / max_value) * (chart_height - 8))
        y0 = chart_bottom - bar_height
        x1 = x0 + bar_width
        draw.rectangle((x0, y0, x1, chart_bottom), fill=_bar_color(index))
        draw.text((x0, chart_bottom + 4), labels[index], fill="black")
        draw.text((x0, max(chart_top, y0 - 14)), f"{value:.3g}", fill="black")


def save_phase2_split_distribution_figure(
    split_metadata: dict,
    output_dir: str | Path = "results/figures/phase2_splits",
) -> Path:
    """Save a compact client-distribution figure for synthetic or real splits."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    clients = split_metadata["clients"]
    if clients and "sample_counts" not in clients[0] and {"split", "client_id", "sample_count"} <= set(clients[0]):
        grouped_clients: dict[int, dict] = {}
        for row in clients:
            client_id = int(row["client_id"])
            grouped = grouped_clients.setdefault(client_id, {"client_id": client_id, "sample_counts": {}})
            grouped["sample_counts"][row["split"]] = int(row["sample_count"])
        clients = [grouped_clients[client_id] for client_id in sorted(grouped_clients)]
    labels = [f"C{client['client_id']}" for client in clients]
    train_samples = [float(client["sample_counts"]["train"]) for client in clients]
    split_mode = split_metadata["split_mode"]
    description = split_metadata["description"]

    if clients and "estimated_object_area_fraction" in clients[0]:
        object_sizes = [float(client["estimated_object_area_fraction"]) for client in clients]
        noise = [float(client["noise_std"]) for client in clients]
        contrast = [float(client["contrast_mean"]) for client in clients]

        canvas = Image.new("RGB", (920, 560), color="white")
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 16), f"Synthetic split distribution: {split_mode}", fill="black")
        draw.text((20, 38), description, fill="black")

        _draw_bar_group(draw, 40, 80, 390, 200, "Train samples per client", train_samples, labels)
        _draw_bar_group(draw, 490, 80, 390, 200, "Estimated object area fraction", object_sizes, labels)
        _draw_bar_group(draw, 40, 320, 390, 200, "Configured noise std", noise, labels)
        _draw_bar_group(draw, 490, 320, 390, 200, "Estimated foreground-background contrast", contrast, labels)
    else:
        val_samples = [float(client["sample_counts"]["val"]) for client in clients]
        test_samples = [float(client["sample_counts"]["test"]) for client in clients]
        total_samples = [
            float(client["sample_counts"]["train"] + client["sample_counts"]["val"] + client["sample_counts"]["test"])
            for client in clients
        ]

        canvas = Image.new("RGB", (920, 560), color="white")
        draw = ImageDraw.Draw(canvas)
        dataset_name = split_metadata.get("dataset", {}).get("dataset", "real_dataset")
        draw.text((20, 16), f"Real dataset split distribution: {dataset_name} / {split_mode}", fill="black")
        draw.text((20, 38), description, fill="black")

        _draw_bar_group(draw, 40, 80, 390, 200, "Train samples per client", train_samples, labels)
        _draw_bar_group(draw, 490, 80, 390, 200, "Validation samples per client", val_samples, labels)
        _draw_bar_group(draw, 40, 320, 390, 200, "Test samples per client", test_samples, labels)
        _draw_bar_group(draw, 490, 320, 390, 200, "Total samples per client", total_samples, labels)

    file_path = output_path / f"{split_mode}_client_distribution.png"
    canvas.save(file_path)
    return file_path


@torch.no_grad()
def save_phase1_prediction_figures(
    model: nn.Module,
    clients: list[FederatedClient],
    output_dir: str | Path = "results/figures/phase1_predictions",
    split: str = "val",
    examples_per_client: int = 2,
) -> list[Path]:
    """Save input/ground-truth/prediction panels for each client."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    saved_paths: list[Path] = []

    for client in clients:
        if split == "val":
            loader = client.val_loader
        elif split == "test":
            loader = client.test_loader
        else:
            raise ValueError("split must be 'val' or 'test'.")

        saved_for_client = 0
        for images, masks in loader:
            images = images.to(client.device)
            masks = masks.to(client.device)
            logits = model(images)
            preds = binary_predictions(logits, from_logits=True)

            for batch_index in range(images.size(0)):
                if saved_for_client >= examples_per_client:
                    break

                image_np = images[batch_index, 0].detach().cpu().numpy()
                mask_np = masks[batch_index, 0].detach().cpu().numpy()
                pred_np = preds[batch_index, 0].detach().cpu().numpy()

                file_path = output_path / f"client{client.client_id}_sample{saved_for_client}_prediction.png"
                _make_tripanel(image_np, mask_np, pred_np).save(file_path)

                saved_paths.append(file_path)
                saved_for_client += 1

            if saved_for_client >= examples_per_client:
                break

    return saved_paths
