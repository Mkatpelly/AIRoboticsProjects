# scripts/validate_closed_loop.py
import argparse
import json
import math
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from learning.policy import build_policy
from scripts.export_policy import sha256
from simulation.environment import make_eval_env

INPUT_NAMES = ("image", "joint_state")


def policy_inputs(obs, expected_shapes):
    feed = {}
    for name in INPUT_NAMES:
        value = np.asarray(obs[name])
        if value.dtype != np.float32 or (1, *value.shape) != expected_shapes[name]:
            raise ValueError(
                f"{name}: expected unbatched float32 "
                f"{expected_shapes[name][1:]}, got {value.dtype} {value.shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"{name}: non-finite observation")
        feed[name] = np.ascontiguousarray(value[None, ...])
    return feed


def to_env_action(output, env):
    # Replace this mapping if training scaled actions before env.step().
    expected = (1, *env.action_space.shape)
    if output.dtype != np.float32 or output.shape != expected:
        raise ValueError(
            f"Expected float32 policy output {expected}, "
            f"got {output.dtype} {output.shape}"
        )
    if not np.isfinite(output).all():
        raise ValueError("Policy returned a non-finite action")

    action = output[0].astype(env.action_space.dtype, copy=True)
    if not env.action_space.contains(action):
        raise ValueError(f"Policy action outside simulator action space: {action}")
    return action


def rollout(env, obs, predict, shapes, max_steps):
    actions = []
    total_reward = 0.0

    for step in range(1, max_steps + 1):
        output = predict(policy_inputs(obs, shapes))
        action = to_env_action(output, env)
        actions.append(action.tolist())

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        success = info.get("is_success")
        if not isinstance(success, (bool, np.bool_)):
            raise ValueError("env.step() must set boolean info['is_success']")

        if terminated or truncated:
            break

    return {
        "success": bool(success),
        "steps": step,
        "return": total_reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "validator_step_limit": not (terminated or truncated),
    }, actions


def write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--max-success-drop-pp", type=float, default=2.0)
    parser.add_argument("--max-disagreements", type=int, default=5)
    args = parser.parse_args()

    if args.episodes < 1 or args.max_steps < 1 or args.seed_start < 0:
        parser.error("episodes and max-steps must be positive; seed-start >= 0")
    if (
        not math.isfinite(args.max_success_drop_pp)
        or not 0 <= args.max_success_drop_pp <= 100
        or args.max_disagreements < 0
    ):
        parser.error("invalid acceptance thresholds")
    if args.out_dir.exists():
        parser.error("--out-dir must be a new directory")

    model_path = args.bundle / "policy.onnx"
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    checkpoint_hash = sha256(args.checkpoint)
    model_hash = sha256(model_path)
    if checkpoint_hash != manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint does not match the export manifest")
    if model_hash != manifest["model_sha256"]:
        raise ValueError("ONNX model does not match the export manifest")

    shapes = {
        name: tuple(manifest["inputs"][name]["shape"]) for name in INPUT_NAMES
    }

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    policy = build_policy().cpu().float().eval()
    policy.load_state_dict(
        checkpoint.get("model_state_dict", checkpoint), strict=True
    )
    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )

    @torch.inference_mode()
    def predict_torch(feed):
        inputs = tuple(torch.from_numpy(feed[name]) for name in INPUT_NAMES)
        return policy(*inputs).cpu().numpy()

    def predict_onnx(feed):
        return session.run(["action"], feed)[0]

    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows = []

    for seed in range(args.seed_start, args.seed_start + args.episodes):
        with ExitStack() as stack:
            torch_env = make_eval_env()
            stack.callback(torch_env.close)
            onnx_env = make_eval_env()
            stack.callback(onnx_env.close)

            torch_obs, _ = torch_env.reset(seed=seed)
            onnx_obs, _ = onnx_env.reset(seed=seed)
            left = policy_inputs(torch_obs, shapes)
            right = policy_inputs(onnx_obs, shapes)
            for name in INPUT_NAMES:
                if not np.array_equal(left[name], right[name]):
                    raise RuntimeError(
                        f"Seed {seed}: initial {name} differs across environments"
                    )

            pt, pt_actions = rollout(
                torch_env, torch_obs, predict_torch, shapes, args.max_steps
            )
            ox, ox_actions = rollout(
                onnx_env, onnx_obs, predict_onnx, shapes, args.max_steps
            )

        rows.append({"seed": seed, "pytorch": pt, "onnx": ox})
        if pt["success"] != ox["success"]:
            write_json(
                args.out_dir / f"disagreement-{seed}.json",
                {
                    "seed": seed,
                    "pytorch": pt,
                    "onnx": ox,
                    "pytorch_actions": pt_actions,
                    "onnx_actions": ox_actions,
                },
            )

    pt_only = sum(r["pytorch"]["success"] and not r["onnx"]["success"] for r in rows)
    ox_only = sum(r["onnx"]["success"] and not r["pytorch"]["success"] for r in rows)
    both = sum(r["pytorch"]["success"] and r["onnx"]["success"] for r in rows)
    n = len(rows)
    drop_pp = 100 * (pt_only - ox_only) / n
    passed = (
        drop_pp <= args.max_success_drop_pp
        and pt_only + ox_only <= args.max_disagreements
    )

    summary = {
        "passed": passed,
        "checkpoint_sha256": checkpoint_hash,
        "model_sha256": model_hash,
        "provider": session.get_providers()[0],
        "seed_start": args.seed_start,
        "max_steps": args.max_steps,
        "episodes": n,
        "pytorch_successes": both + pt_only,
        "onnx_successes": both + ox_only,
        "both_succeed": both,
        "both_fail": n - both - pt_only - ox_only,
        "pytorch_only_succeeds": pt_only,
        "onnx_only_succeeds": ox_only,
        "success_drop_percentage_points": drop_pp,
        "disagreements": pt_only + ox_only,
        "max_success_drop_pp": args.max_success_drop_pp,
        "max_disagreements": args.max_disagreements,
    }
    write_json(args.out_dir / "report.json", {"summary": summary, "episodes": rows})
    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
