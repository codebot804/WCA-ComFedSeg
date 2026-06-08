"""Validate Phase 5A real dataset folders without training."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.busi import discover_busi_samples
from datasets.kvasir_seg import discover_kvasir_samples

REPORT_PATH = PROJECT_ROOT / "results/summaries/phase5a_dataset_validation.md"


def _status_lines(status: dict[str, Any]) -> list[str]:
    return [f"- `{key}`: {value}" for key, value in status.items()]


def _warning_lines(warnings: list[str]) -> list[str]:
    if not warnings:
        return ["- None"]
    return [f"- {warning}" for warning in warnings]


def build_report() -> tuple[str, dict[str, Any]]:
    busi_samples, busi_metadata = discover_busi_samples()
    kvasir_samples, kvasir_metadata = discover_kvasir_samples()

    lines = [
        "# Phase 5A Dataset Validation",
        "",
        "This report validates local real dataset files only. It does not train models and does not create paper evidence.",
        "",
        "## BUSI",
        "",
        f"- Root: `{busi_metadata['root']}`",
        "- Folder status:",
        *_status_lines(busi_metadata["folder_status"]),
        f"- Valid image-mask pair count: {len(busi_samples)}",
        f"- Benign valid pairs: {busi_metadata['class_counts'].get('benign', 0)}",
        f"- Malignant valid pairs: {busi_metadata['class_counts'].get('malignant', 0)}",
        f"- Images with multiple masks: {busi_metadata['images_with_multiple_masks']}",
        f"- Extra masks merged: {busi_metadata['merged_mask_count']}",
        f"- Unmatched images: {busi_metadata['unmatched_image_count']}",
        f"- Orphan masks: {busi_metadata['orphan_mask_count']}",
        "- Warnings:",
        *_warning_lines(busi_metadata["warnings"]),
        "",
        "## Kvasir-SEG",
        "",
        f"- Root: `{kvasir_metadata['root']}`",
        "- Folder status:",
        *_status_lines(kvasir_metadata["folder_status"]),
        f"- Valid image-mask pair count: {len(kvasir_samples)}",
        f"- Raw image count: {kvasir_metadata['raw_image_count']}",
        f"- Raw mask count: {kvasir_metadata['raw_mask_count']}",
        f"- Unmatched images: {kvasir_metadata['unmatched_image_count']}",
        f"- Unmatched masks: {kvasir_metadata['unmatched_mask_count']}",
        "- Warnings:",
        *_warning_lines(kvasir_metadata["warnings"]),
        "",
        "## Phase 5A Readiness",
        "",
        f"- BUSI valid for smoke testing: {len(busi_samples) > 0 and not busi_metadata['warnings']}",
        f"- Kvasir-SEG valid for smoke testing: {len(kvasir_samples) > 0 and not kvasir_metadata['warnings']}",
    ]
    payload = {"busi": busi_metadata, "kvasir_seg": kvasir_metadata}
    return "\n".join(lines) + "\n", payload


def main() -> None:
    report, payload = build_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Dataset validation report saved to: {REPORT_PATH}")
    print(f"BUSI valid pairs: {payload['busi']['valid_pair_count']}")
    print(f"Kvasir-SEG valid pairs: {payload['kvasir_seg']['valid_pair_count']}")


if __name__ == "__main__":
    main()
