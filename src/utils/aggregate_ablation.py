"""Aggregate ablation CSV results into a Markdown table.

Usage:
    python -m src.utils.aggregate_ablation --csv ablation/results/all_runs.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


DISPLAY_NAMES = {
    "baseline_3d_unet": "3D U-Net baseline",
    "full_model": "Full model",
    "ablation_A1_no_convlstm": "A1 - No ConvLSTM",
    "ablation_A2_no_pretraining": "A2 - No autoencoder pretraining",
    "ablation_A3_no_attention": "A3 - No attention gates",
    "ablation_A4_no_vae": "A4 - No VAE branch",
    "ablation_A7_no_augmentation": "A7 - No augmentation",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def best_by_scenario(rows: list[dict[str, str]]) -> list[tuple[str, float]]:
    best: dict[str, float] = {}
    for row in rows:
        scenario = row["scenario"]
        dice = float(row["best_val_dice_mean"])
        best[scenario] = max(best.get(scenario, 0.0), dice)
    return sorted(best.items(), key=lambda item: item[1], reverse=True)


def to_markdown(results: list[tuple[str, float]]) -> str:
    lines = ["| Scenario | Best validation mean Dice |", "|---|---:|"]
    for scenario, dice in results:
        name = DISPLAY_NAMES.get(scenario, scenario)
        lines.append(f"| {name} | {dice:.4f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="ablation/results/all_runs.csv")
    args = parser.parse_args()
    rows = read_rows(Path(args.csv))
    print(to_markdown(best_by_scenario(rows)))


if __name__ == "__main__":
    main()

