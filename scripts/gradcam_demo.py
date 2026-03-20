from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chexpert_poc.common.config import get_config_bool, get_section, load_config
from chexpert_poc.common.io import ensure_dir, save_json
from chexpert_poc.common.runtime import get_device
from chexpert_poc.common.model_config import resolve_num_classes

from chexpert_poc.datasets.chexpert_dataset import build_image_transform

from chexpert_poc.inference.checkpoint import (
    load_checkpoint,
    resolve_checkpoint_path,
    validate_checkpoint_config,
)
from chexpert_poc.inference.gradcam_service import (
    generate_gradcam_result,
    save_gradcam_artifacts,
)
from chexpert_poc.inference.postprocess import load_thresholds
from chexpert_poc.inference.predictor import build_model_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="single image path")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="explicit checkpoint path; if omitted, latest best.pt is used",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Grad-CAM을 만들 라벨 이름. 없으면 최고 확률 라벨 사용",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="comma-separated thresholds, e.g. 0.47,0.22,0.36,0.39,0.60",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.75,
        help="overlay 강도 (0~1)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    data_cfg = get_section(config, "data")
    paths_cfg = get_section(config, "paths")

    label_names = list(data_cfg["target_labels"])
    image_size = int(data_cfg["image_size"])
    channels_last = get_config_bool(config, "train", "channels_last", default=True)
    num_classes = resolve_num_classes(config)

    checkpoint_path = resolve_checkpoint_path(
        output_root=paths_cfg["output_root"],
        checkpoint=args.checkpoint,
    )
    checkpoint_meta = load_checkpoint(checkpoint_path)
    validate_checkpoint_config(checkpoint_meta, config)

    thresholds, threshold_source = load_thresholds(
        checkpoint_path=checkpoint_path,
        label_names=label_names,
        thresholds_arg=args.thresholds,
    )

    transform = build_image_transform(image_size=image_size)

    model = build_model_from_checkpoint(
        checkpoint=checkpoint_meta,
        num_classes=num_classes,
        device=device,
        channels_last=channels_last,
    )

    gradcam_result = generate_gradcam_result(
        model=model,
        image_path=args.input,
        transform=transform,
        device=device,
        label_names=label_names,
        thresholds=thresholds,
        channels_last=channels_last,
        target_label_arg=args.label,
        alpha=float(args.alpha),
    )

    run_name = datetime.now().strftime("gradcam_%Y%m%d_%H%M%S")
    output_dir = ensure_dir(Path(paths_cfg["output_root"]) / "gradcam_runs" / run_name)

    artifacts = save_gradcam_artifacts(
        result=gradcam_result,
        output_dir=output_dir,
    )

    result_json = {
        "input_path": gradcam_result["input_path"],
        "checkpoint_path": str(checkpoint_path),
        "threshold_source": threshold_source,
        "target_label": gradcam_result["target_label"],
        "target_prob": gradcam_result["target_prob"],
        "positive_labels": gradcam_result["positive_labels"],
        "predictions": gradcam_result["predictions"],
        "artifacts": artifacts,
    }

    save_json(result_json, output_dir / "gradcam_result.json")

    print("=" * 100)
    print("gradcam_demo.py start")
    print("=" * 100)
    print(f"device          : {device}")
    print(f"checkpoint_path : {checkpoint_path}")
    print(f"threshold_source: {threshold_source}")
    print(f"input_path      : {gradcam_result['input_path']}")
    print(f"target_label    : {gradcam_result['target_label']}")
    print(f"target_prob     : {gradcam_result['target_prob']:.4f}")
    print(f"positive_labels : {gradcam_result['positive_labels']}")

    print("\n[predictions]")
    for pred in gradcam_result["predictions"]:
        print(
            f"- {pred['label']}: "
            f"prob={pred['prob']:.4f}, "
            f"threshold={pred['threshold']:.2f}, "
            f"pred={pred['pred']}"
        )

    print("\nartifacts saved to:")
    print(output_dir)
    print("=" * 100)


if __name__ == "__main__":
    main()