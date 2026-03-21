from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from chexpert_poc.common.config import get_config_bool, get_section
from chexpert_poc.common.io import ensure_dir, save_json
from chexpert_poc.common.model_config import resolve_num_classes
from chexpert_poc.common.runtime import get_device
from chexpert_poc.datasets.chexpert_dataset import build_image_transform

from chexpert_poc.inference.artifact_io import save_inference_predictions_csv
from chexpert_poc.inference.checkpoint import (
    load_checkpoint,
    resolve_checkpoint_path,
    validate_checkpoint_config,
)
from chexpert_poc.inference.gradcam_service import (
    generate_gradcam_result,
    save_gradcam_artifacts,
)
from chexpert_poc.inference.input_io import validate_input_image
from chexpert_poc.inference.postprocess import load_thresholds
from chexpert_poc.inference.predictor import (
    build_model_from_checkpoint,
    predict_one_image,
)


def build_inference_context(
    *,
    config: dict[str, Any],
    checkpoint: str | None = None,
    thresholds_arg: str | None = None,
) -> dict[str, Any]:
    """
    서비스/CLI 공용 추론 실행 컨텍스트를 구성한다.

    포함:
    - device
    - label_names
    - thresholds
    - checkpoint_path
    - transform
    - model
    """
    device = get_device()

    data_cfg = get_section(config, "data")
    paths_cfg = get_section(config, "paths")

    label_names = list(data_cfg["target_labels"])
    image_size = int(data_cfg["image_size"])
    channels_last = get_config_bool(config, "train", "channels_last", default=True)
    num_classes = resolve_num_classes(config)

    checkpoint_path = resolve_checkpoint_path(
        output_root=paths_cfg["output_root"],
        checkpoint=checkpoint,
    )
    checkpoint_meta = load_checkpoint(checkpoint_path)
    validate_checkpoint_config(checkpoint_meta, config)

    thresholds, threshold_source = load_thresholds(
        checkpoint_path=checkpoint_path,
        label_names=label_names,
        thresholds_arg=thresholds_arg,
    )

    transform = build_image_transform(image_size=image_size)

    model = build_model_from_checkpoint(
        checkpoint=checkpoint_meta,
        num_classes=num_classes,
        device=device,
        channels_last=channels_last,
    )

    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "threshold_source": threshold_source,
        "thresholds": thresholds,
        "labels": label_names,
        "num_classes": num_classes,
        "device": str(device),
        "channels_last": channels_last,
        "pretrained": False,
    }

    return {
        "device": device,
        "label_names": label_names,
        "image_size": image_size,
        "channels_last": channels_last,
        "num_classes": num_classes,
        "checkpoint_path": checkpoint_path,
        "checkpoint_meta": checkpoint_meta,
        "thresholds": thresholds,
        "threshold_source": threshold_source,
        "transform": transform,
        "model": model,
        "output_root": Path(paths_cfg["output_root"]),
        "metadata": metadata,
    }


def predict_single_image_service(
    *,
    config: dict[str, Any],
    image_path: str,
    checkpoint: str | None = None,
    thresholds_arg: str | None = None,
    save_outputs: bool = False,
) -> dict[str, Any]:
    """
    단일 이미지 추론 서비스용 엔트리포인트.

    save_outputs=False:
    - 메모리 상 결과만 반환

    save_outputs=True:
    - infer_runs 아래에 json/csv 저장 후 경로까지 반환
    """
    context = build_inference_context(
        config=config,
        checkpoint=checkpoint,
        thresholds_arg=thresholds_arg,
    )

    image_path_resolved = validate_input_image(image_path)

    prediction = predict_one_image(
        model=context["model"],
        image_path=image_path_resolved,
        transform=context["transform"],
        device=context["device"],
        label_names=context["label_names"],
        thresholds=context["thresholds"],
        channels_last=context["channels_last"],
    )

    result = {
        "mode": "single_image_inference",
        "input_path": str(image_path_resolved),
        "metadata": {
            **context["metadata"],
            "num_inputs": 1,
            "recursive": False,
        },
        "prediction": prediction,
    }

    if save_outputs:
        run_name = datetime.now().strftime("service_infer_%Y%m%d_%H%M%S")
        output_dir = ensure_dir(context["output_root"] / "infer_runs" / run_name)

        save_json(result["metadata"], output_dir / "infer_metadata.json")
        save_json([prediction], output_dir / "predictions.json")
        save_inference_predictions_csv(
            predictions=[prediction],
            output_csv_path=output_dir / "predictions.csv",
            label_names=context["label_names"],
        )

        result["artifacts"] = {
            "output_dir": str(output_dir),
            "metadata_json": str(output_dir / "infer_metadata.json"),
            "predictions_json": str(output_dir / "predictions.json"),
            "predictions_csv": str(output_dir / "predictions.csv"),
        }

    return result


def predict_single_image_with_gradcam_service(
    *,
    config: dict[str, Any],
    image_path: str,
    checkpoint: str | None = None,
    thresholds_arg: str | None = None,
    label_arg: str | None = None,
    alpha: float = 0.75,
    save_outputs: bool = False,
) -> dict[str, Any]:
    """
    단일 이미지 + Grad-CAM 서비스용 엔트리포인트.

    save_outputs=False:
    - 메타데이터/예측 결과만 반환

    save_outputs=True:
    - gradcam_runs 아래에 이미지/json 저장 후 경로 반환
    """
    context = build_inference_context(
        config=config,
        checkpoint=checkpoint,
        thresholds_arg=thresholds_arg,
    )

    gradcam_result = generate_gradcam_result(
        model=context["model"],
        image_path=image_path,
        transform=context["transform"],
        device=context["device"],
        label_names=context["label_names"],
        thresholds=context["thresholds"],
        channels_last=context["channels_last"],
        target_label_arg=label_arg,
        alpha=float(alpha),
    )

    result = {
        "mode": "single_image_gradcam",
        "input_path": gradcam_result["input_path"],
        "metadata": {
            **context["metadata"],
            "num_inputs": 1,
            "recursive": False,
        },
        "target_label": gradcam_result["target_label"],
        "target_prob": gradcam_result["target_prob"],
        "positive_labels": gradcam_result["positive_labels"],
        "predictions": gradcam_result["predictions"],
    }

    if save_outputs:
        run_name = datetime.now().strftime("service_gradcam_%Y%m%d_%H%M%S")
        output_dir = ensure_dir(context["output_root"] / "gradcam_runs" / run_name)

        artifacts = save_gradcam_artifacts(
            result=gradcam_result,
            output_dir=output_dir,
        )

        result_with_artifacts = {
            **result,
            "artifacts": artifacts,
        }

        save_json(result_with_artifacts, output_dir / "gradcam_result.json")

        result["artifacts"] = {
            **artifacts,
            "output_dir": str(output_dir),
            "result_json": str(output_dir / "gradcam_result.json"),
        }

    return result