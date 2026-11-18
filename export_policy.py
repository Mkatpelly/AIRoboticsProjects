# scripts/export_policy.py
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import onnxruntime as ort
import torch

from learning.policy import build_policy  # Use your training-time factory.

INPUT_NAMES = ["image", "joint_state"]
OUTPUT_NAME = "action"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fixture(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        images = np.ascontiguousarray(data["image"])
        joints = np.ascontiguousarray(data["joint_state"])

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("image must have shape [N, 3, H, W] (RGB, NCHW)")
    if joints.ndim != 2 or joints.shape[0] != images.shape[0]:
        raise ValueError("joint_state must have shape [N, J] with matching N")

    for name, array in zip(INPUT_NAMES, (images, joints)):
        if not array.size or array.dtype != np.float32:
            raise ValueError(f"{name} must be nonempty and float32")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or infinity")

    return images, joints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=None)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    if any(not math.isfinite(t) or t < 0 for t in (args.atol, args.rtol)):
        parser.error("--atol and --rtol must be finite and nonnegative")
    if args.out_dir.exists():
        parser.error("--out-dir must be a new directory")

    images, joints = load_fixture(args.fixture)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    policy = build_policy().cpu().float().eval()
    policy.load_state_dict(state_dict, strict=True)

    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out_dir.parent) as temp_dir:
        bundle = Path(temp_dir) / "bundle"
        bundle.mkdir()
        model_path = bundle / "policy.onnx"

        example_inputs = (
            torch.from_numpy(images[:1]),
            torch.from_numpy(joints[:1]),
        )
        with torch.no_grad():
            torch.onnx.export(
                policy,
                example_inputs,
                str(model_path),
                input_names=INPUT_NAMES,
                output_names=[OUTPUT_NAME],
                dynamo=True,
                external_data=False,
                opset_version=args.opset,
            )

        onnx.checker.check_model(str(model_path), full_check=True)
        graph = onnx.load(str(model_path))
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )

        expected_shapes = [(1, *images.shape[1:]), (1, *joints.shape[1:])]
        actual_inputs = [
            (item.name, tuple(item.shape), item.type)
            for item in session.get_inputs()
        ]
        expected_inputs = [
            (name, shape, "tensor(float)")
            for name, shape in zip(INPUT_NAMES, expected_shapes)
        ]
        if actual_inputs != expected_inputs:
            raise RuntimeError(f"Unexpected ONNX input contract: {actual_inputs}")

        outputs = session.get_outputs()
        if len(outputs) != 1 or outputs[0].name != OUTPUT_NAME:
            raise RuntimeError("Expected exactly one ONNX output named 'action'")

        max_error_by_action = None
        with torch.inference_mode():
            for i in range(len(images)):
                feed = {"image": images[i:i + 1], "joint_state": joints[i:i + 1]}
                reference = policy(
                    torch.from_numpy(feed["image"]),
                    torch.from_numpy(feed["joint_state"]),
                ).cpu().numpy()
                actual = session.run([OUTPUT_NAME], feed)[0]

                if (
                    reference.ndim != 2
                    or reference.shape[0] != 1
                    or reference.shape[1] == 0
                    or actual.shape != reference.shape
                    or tuple(outputs[0].shape) != reference.shape
                ):
                    raise RuntimeError(f"Sample {i}: expected action shape [1, A]")
                if actual.dtype != np.float32 or reference.dtype != np.float32:
                    raise RuntimeError(f"Sample {i}: expected FP32 actions")
                if not (np.isfinite(reference).all() and np.isfinite(actual).all()):
                    raise RuntimeError(f"Sample {i}: non-finite action")

                np.testing.assert_allclose(
                    actual,
                    reference,
                    atol=args.atol,
                    rtol=args.rtol,
                    equal_nan=False,
                    err_msg=f"Action parity failed at fixture sample {i}",
                )
                error = np.abs(actual[0].astype(np.float64) - reference[0])
                max_error_by_action = (
                    error if max_error_by_action is None
                    else np.maximum(max_error_by_action, error)
                )

        manifest = {
            "checkpoint_sha256": sha256(args.checkpoint),
            "fixture_sha256": sha256(args.fixture),
            "model_sha256": sha256(model_path),
            "provider": session.get_providers()[0],
            "versions": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "onnx": onnx.__version__,
                "onnxruntime": ort.__version__,
                "numpy": np.__version__,
            },
            "opsets": {item.domain or "ai.onnx": item.version
                       for item in graph.opset_import},
            "inputs": {
                name: {"shape": list(shape), "dtype": "float32"}
                for name, shape in zip(INPUT_NAMES, expected_shapes)
            },
            "output": {
                "name": OUTPUT_NAME,
                "shape": list(outputs[0].shape),
                "dtype": "float32",
            },
            "parity": {
                "samples": len(images),
                "atol": args.atol,
                "rtol": args.rtol,
                "max_abs_error_by_action": max_error_by_action.tolist(),
            },
        }
        (bundle / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        bundle.rename(args.out_dir)

    print(f"Exported policy and validated {len(images)} observations: {args.out_dir}")


if __name__ == "__main__":
    main()
