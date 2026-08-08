"""
buff_route_speed.py

Scale up the recorded speed in a manual_lap_*.npz (used by record_manual_lap.py's
--route-npz autopilot mode) so the route-follower drives faster than the original
recording, while capping the top speed so corners don't get an unrealistic target
that would send the car into a wall.
"""
import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_npz", help="A manual_lap_*.npz recording")
    parser.add_argument("--scale", type=float, default=1.35, help="Multiply recorded speed by this factor")
    parser.add_argument("--max-mph", type=float, default=115.0, help="Hard cap on the scaled speed")
    parser.add_argument("--out", type=str, default=None, help="Output path (default: <input>_buffed.npz)")
    args = parser.parse_args()

    data = dict(np.load(args.input_npz))
    max_mps = args.max_mph / 2.23694

    original_speed = data["speed"]
    buffed_speed = np.clip(original_speed * args.scale, 0.0, max_mps)

    # Keep vel_x/vel_y/vel_z consistent with the new speed magnitude (same direction, scaled length)
    velocity_norm = np.linalg.norm(np.stack([data["vel_x"], data["vel_y"], data["vel_z"]], axis=1), axis=1)
    ratio = np.divide(buffed_speed, velocity_norm, out=np.ones_like(buffed_speed), where=velocity_norm > 1e-6)
    data["vel_x"] = data["vel_x"] * ratio
    data["vel_y"] = data["vel_y"] * ratio
    data["vel_z"] = data["vel_z"] * ratio
    data["speed"] = buffed_speed

    out_path = args.out or args.input_npz.replace(".npz", "_buffed.npz")
    np.savez_compressed(out_path, **data)

    print(f"Saved buffed route to {out_path}")
    print(f"Speed range before (mph): {original_speed.min() * 2.23694:.1f} - {original_speed.max() * 2.23694:.1f}")
    print(f"Speed range after  (mph): {buffed_speed.min() * 2.23694:.1f} - {buffed_speed.max() * 2.23694:.1f}")


if __name__ == "__main__":
    main()
