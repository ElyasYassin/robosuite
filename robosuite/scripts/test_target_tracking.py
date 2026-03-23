"""
Quick smoke-test script for the TargetTracking environment.

Examples:
    # On-screen visualization with random actions
    python robosuite/scripts/test_target_tracking.py --onscreen

    # Offscreen eye-in-hand camera observations (prints image keys / shapes)
    python robosuite/scripts/test_target_tracking.py --camera-obs --camera all-eye_in_hand
"""

import argparse
import time

import numpy as np

import robosuite as suite


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test TargetTracking with UR5e")
    parser.add_argument("--steps", type=int, default=300, help="Max steps to run")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--onscreen",
        action="store_true",
        help="Enable on-screen rendering window",
    )
    parser.add_argument(
        "--camera-obs",
        action="store_true",
        help="Enable image observations from camera_names",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="all-eye_in_hand",
        help="Camera name pattern for observations (e.g. all-eye_in_hand, agentview)",
    )
    parser.add_argument("--width", type=int, default=128, help="Camera width")
    parser.add_argument("--height", type=int, default=128, help="Camera height")
    parser.add_argument(
        "--moving-target",
        action="store_true",
        help="If set, target moves during episode",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional per-step sleep to slow down playback (seconds)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    env = suite.make(
        env_name="TargetTracking",
        robots="UR5e",
        render_camera="robot0_eye_in_hand",  # try "eye_in_hand" if this fails
        has_renderer=args.onscreen,
        has_offscreen_renderer=args.camera_obs,
        use_camera_obs=args.camera_obs,
        use_object_obs=True,
        camera_names=args.camera,
        camera_widths=args.width,
        camera_heights=args.height,
        reward_shaping=True,
        moving_target=args.moving_target,
        horizon=args.steps,
        seed=args.seed,
    )

    try:
        obs = env.reset()
        print(f"Reset complete. Observation keys ({len(obs)} total):")
        print(sorted(obs.keys()))

        image_keys = [k for k in obs if k.endswith("_image")]
        if image_keys:
            print("Camera image keys + shapes:")
            for k in image_keys:
                print(f"  {k}: {obs[k].shape}")

        low, high = env.action_spec
        print(f"Action dim: {low.shape[0]}")
        print(f"Action low[:7]: {np.array2string(low[:7], precision=3)}")
        print(f"Action high[:7]: {np.array2string(high[:7], precision=3)}")

        ep_return = 0.0
        for t in range(args.steps):
            action = np.random.uniform(low, high)
            obs, reward, done, info = env.step(action)
            ep_return += reward

            if args.onscreen:
                env.render()
                if args.sleep > 0:
                    time.sleep(args.sleep)

            if t % 25 == 0 or done:
                dist = np.linalg.norm(obs["eef_to_target"]) if "eef_to_target" in obs else float("nan")
                print(f"t={t:04d} reward={reward:.4f} dist={dist:.4f} done={done}")

            if done:
                break

        print(f"Finished. Episode return: {ep_return:.4f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

