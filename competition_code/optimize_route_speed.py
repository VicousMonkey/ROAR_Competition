"""
optimize_route_speed.py

Automatically tune a per-waypoint speed profile (sourced from a manual_lap_*.npz
recording) as high as possible without crashing, by actually driving it in CARLA
via telemetry (collision sensor + off-track lateral distance) instead of relying
on a human watching the screen.

Each iteration drives the full route with the current speed profile:
  - If it fails (collision, or the car drifts off the track surface), the speed
    is scaled DOWN only in a window around the failure point, and retried.
  - If it succeeds cleanly, the whole profile is scaled UP a bit and retried,
    to keep pushing toward the fastest profile that still holds the road.

Saves the best known-safe profile after every iteration (not just at the end),
so progress isn't lost if you stop it early.
"""
import argparse
import asyncio
import contextlib
import io

import carla
import numpy as np
import roar_py_carla
import roar_py_interface

from submission import filter_waypoints, normalize_rad

COLLISION_IMPULSE_THRESHOLD = 100.0
OFFTRACK_MARGIN_FACTOR = 1.0   # fraction of half lane-width allowed before flagging "about to hit the wall"
                                # (measured a real, self-correcting launch wobble peak at ~6.67m against a 6.0m
                                # half lane-width with zero collision - the true wall tolerance is looser than
                                # lane_width metadata suggests, so the collision sensor is the real ground truth)
OFFTRACK_DEBOUNCE_TICKS = 60   # must be off-track this many consecutive ticks (~3s) to count as a real failure,
                                # not a brief self-correcting wobble
MAX_STEER_DELTA = 0.05
MAX_THROTTLE_DELTA = 0.08
MAX_BRAKE_DELTA = 0.12


def build_route_from_recording(npz_path: str):
    data = np.load(npz_path)
    waypoint_idx = data["waypoint_idx"]
    locations = np.stack([data["loc_x"], data["loc_y"], data["loc_z"]], axis=1)
    speeds = data["speed"]

    order, loc_by_idx, speed_by_idx = [], {}, {}
    for i in range(len(waypoint_idx)):
        idx = int(waypoint_idx[i])
        if idx not in loc_by_idx:
            loc_by_idx[idx] = locations[i]
            order.append(idx)
        speed_by_idx[idx] = max(speed_by_idx.get(idx, 0.0), float(speeds[i]))

    locs = np.array([loc_by_idx[i] for i in order])
    spds = np.array([speed_by_idx[i] for i in order])
    return locs, spds


def make_waypoints(locs: np.ndarray):
    return [roar_py_interface.RoarPyWaypoint(loc, np.zeros(3), 8.0) for loc in locs]


async def run_attempt(roar_py_instance, world, track_waypoints, route_waypoints, route_speeds, max_ticks: int):
    vehicle = None
    for spawn_attempt in range(5):
        vehicle = world.spawn_vehicle(
            "vehicle.audi.tt",
            route_waypoints[0].location + np.array([0, 0, 1]),
            track_waypoints[0].roll_pitch_yaw,
            True,
        )
        if vehicle is not None:
            break
        print(f"spawn_vehicle returned None (attempt {spawn_attempt + 1}/5), "
              f"cleaning up stale actors and retrying...", flush=True)
        roar_py_instance.clean_actors_not_registered()
        for _ in range(5):
            await world.step()
    if vehicle is None:
        raise RuntimeError("spawn_vehicle kept returning None after 5 retries - CARLA may need a restart")

    location_sensor = vehicle.attach_location_in_world_sensor()
    velocity_sensor = vehicle.attach_velocimeter_sensor()
    rpy_sensor = vehicle.attach_roll_pitch_yaw_sensor()
    collision_sensor = vehicle.attach_collision_sensor(np.zeros(3), np.zeros(3))

    for _ in range(20):
        await world.step()
    await vehicle.receive_observation()

    # True nearest-neighbor lookup for the off-track check, precomputed once.
    # filter_waypoints() is a forward-search designed for a vehicle hugging its own
    # waypoint sequence; it stalls (false "off-track") once the car's line drifts a
    # few meters from the track's official waypoints, which is normal human driving.
    track_locations_2d = np.array([wp.location[:2] for wp in track_waypoints])
    track_lane_widths = np.array([wp.lane_width for wp in track_waypoints])

    route_idx = 0
    applied_steer = applied_throttle = applied_brake = 0.0
    consecutive_offtrack_ticks = 0
    result = {"status": "timeout", "fail_route_idx": None, "ticks": 0, "furthest_route_idx": 0}

    try:
        for tick in range(max_ticks):
            await vehicle.receive_observation()
            loc = location_sensor.get_last_gym_observation()
            rot = rpy_sensor.get_last_gym_observation()
            vel = velocity_sensor.get_last_gym_observation()
            speed = float(np.linalg.norm(vel))

            route_idx = filter_waypoints(loc, route_idx, route_waypoints)

            dists_to_track = np.linalg.norm(track_locations_2d - loc[:2], axis=1)
            nearest_track_i = int(np.argmin(dists_to_track))
            lateral_dist = float(dists_to_track[nearest_track_i])
            # A brief spike (e.g. a launch wobble that self-corrects) shouldn't count as a
            # failure - only a SUSTAINED off-track condition (OFFTRACK_DEBOUNCE_TICKS in a
            # row) is treated as actually leaving the track surface.
            if lateral_dist > track_lane_widths[nearest_track_i] / 2 * OFFTRACK_MARGIN_FACTOR:
                consecutive_offtrack_ticks += 1
            else:
                consecutive_offtrack_ticks = 0
            if consecutive_offtrack_ticks >= OFFTRACK_DEBOUNCE_TICKS:
                result["status"] = "offtrack"
                result["fail_route_idx"] = route_idx
                break

            collision_impulse = float(np.linalg.norm(collision_sensor.get_last_observation().impulse_normal))
            if collision_impulse > COLLISION_IMPULSE_THRESHOLD:
                result["status"] = "collision"
                result["fail_route_idx"] = route_idx
                break

            target_idx = (route_idx + 3) % len(route_waypoints)
            target = route_waypoints[target_idx]
            vector_to_target = (target.location - loc)[:2]
            heading_to_target = np.arctan2(vector_to_target[1], vector_to_target[0])
            delta_heading = normalize_rad(heading_to_target - rot[2])
            # Floor the speed used in the steer gain instead of switching to a sign()
            # bang-bang controller at low speed: that discontinuity snaps to full lock
            # for *any* nonzero heading error (even sensor noise) while stationary,
            # which was causing the car to launch already at full steer and swerve off.
            steer_speed_floor = max(speed, 2.0)
            steer_target = -8.0 / np.sqrt(steer_speed_floor) * delta_heading / np.pi
            steer_target = float(np.clip(steer_target, -1.0, 1.0))

            target_speed = float(route_speeds[target_idx])
            throttle_raw = 0.05 * (target_speed - speed)
            throttle_target = float(np.clip(throttle_raw, 0.0, 1.0))
            brake_target = float(np.clip(-throttle_raw, 0.0, 1.0))

            applied_steer += float(np.clip(steer_target - applied_steer, -MAX_STEER_DELTA, MAX_STEER_DELTA))
            applied_throttle += float(np.clip(throttle_target - applied_throttle, -MAX_THROTTLE_DELTA, MAX_THROTTLE_DELTA))
            applied_brake += float(np.clip(brake_target - applied_brake, -MAX_BRAKE_DELTA, MAX_BRAKE_DELTA))

            control = {
                "throttle": applied_throttle, "steer": applied_steer, "brake": applied_brake,
                "hand_brake": np.array([0]), "reverse": np.array([0]), "target_gear": 0,
            }
            await vehicle.apply_action(control)
            await world.step()
            result["ticks"] = tick
            result["furthest_route_idx"] = route_idx

            if route_idx >= len(route_waypoints) - 5:
                result["status"] = "success"
                break
    finally:
        vehicle.close()

    return result


def save_route(out_path: str, locs: np.ndarray, speed_profile: np.ndarray):
    n = len(locs)
    np.savez_compressed(
        out_path,
        waypoint_idx=np.arange(n),
        loc_x=locs[:, 0], loc_y=locs[:, 1], loc_z=locs[:, 2],
        speed=speed_profile,
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_npz")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--max-ticks", type=int, default=4000)
    parser.add_argument("--scale-up", type=float, default=1.08, help="Global speed multiplier applied after a clean run")
    parser.add_argument("--scale-down", type=float, default=0.85, help="Local speed multiplier applied around a failure point")
    parser.add_argument("--window", type=int, default=12, help="How many waypoints on each side of a failure to scale down")
    parser.add_argument("--max-mph", type=float, default=130.0)
    parser.add_argument("--min-mph", type=float, default=8.0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out_path = args.out or args.input_npz.replace(".npz", "_optimized.npz")
    progress_path = args.out.replace(".npz", "_progress.npz") if args.out else args.input_npz.replace(".npz", "_progress.npz")
    max_mps = args.max_mph / 2.23694
    min_mps = args.min_mph / 2.23694

    locs, speed_profile = build_route_from_recording(args.input_npz)
    route_waypoints = make_waypoints(locs)
    print(f"Loaded {len(route_waypoints)} route points, starting avg {speed_profile.mean()*2.23694:.1f} mph, "
          f"max {speed_profile.max()*2.23694:.1f} mph", flush=True)

    carla_client = carla.Client('127.0.0.1', 2000)
    carla_client.set_timeout(10.0)
    roar_py_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)
    world = roar_py_instance.world
    world.set_control_steps(0.05, 0.005)
    world.set_asynchronous(False)
    track_waypoints = world.maneuverable_waypoints

    best_profile = None
    best_avg_mph = None
    any_success = False

    try:
        for iteration in range(1, args.iterations + 1):
            print(f"\n=== Iteration {iteration}/{args.iterations} | "
                  f"avg {speed_profile.mean()*2.23694:.1f} mph, max {speed_profile.max()*2.23694:.1f} mph ===", flush=True)

            with contextlib.redirect_stdout(io.StringIO()):
                result = await run_attempt(roar_py_instance, world, track_waypoints, route_waypoints, speed_profile, args.max_ticks)

            print(f"Result: {result['status']} at tick {result['ticks']}"
                  f"{', route idx ' + str(result['fail_route_idx']) if result['fail_route_idx'] is not None else ''}", flush=True)

            if result["status"] == "success":
                any_success = True
                best_profile = speed_profile.copy()
                best_avg_mph = speed_profile.mean() * 2.23694
                save_route(out_path, locs, best_profile)
                print(f"Clean lap! Saved to {out_path} (avg {best_avg_mph:.1f} mph). Pushing speed up for next attempt.", flush=True)
                speed_profile = np.clip(speed_profile * args.scale_up, min_mps, max_mps)
            elif result["status"] == "timeout":
                # Didn't crash or go off-track anywhere - it just ran out of tick budget.
                # That's not a location-specific failure, so don't punish any waypoint for it.
                print(f"Ran out of ticks without crashing (reached route idx {result['furthest_route_idx']} "
                      f"of {len(route_waypoints)}) - raise --max-ticks to let it actually finish. "
                      f"Speed profile left unchanged.", flush=True)
            else:
                fail_idx = result["fail_route_idx"] if result["fail_route_idx"] is not None else 0
                lo = max(0, fail_idx - args.window)
                hi = min(len(speed_profile), fail_idx + args.window)
                speed_profile[lo:hi] = np.clip(speed_profile[lo:hi] * args.scale_down, min_mps, max_mps)
                print(f"Backed off speed in waypoints [{lo}, {hi}) around the failure point.", flush=True)
                # revert everywhere else back to the best known-safe profile so we don't compound
                # unrelated regressions from earlier failed attempts
                if best_profile is not None:
                    unchanged_mask = np.ones(len(speed_profile), dtype=bool)
                    unchanged_mask[lo:hi] = False
                    speed_profile[unchanged_mask] = best_profile[unchanged_mask]

            # Save whatever the current working profile is after every iteration (even without
            # a full clean lap yet), so a crash in the harness itself doesn't lose real progress
            # (e.g. corners already cleared further into the lap).
            save_route(progress_path, locs, speed_profile)

    finally:
        roar_py_instance.close()

    if any_success:
        save_route(out_path, locs, best_profile)
        print(f"\nDone. Best safe profile: avg {best_avg_mph:.1f} mph, max {best_profile.max()*2.23694:.1f} mph.", flush=True)
        print(f"Saved to {out_path}", flush=True)
    else:
        print(f"\nDone. No iteration ever completed a full clean lap - nothing was saved. "
              f"The failures weren't fixed by lowering speed, so this points to a control/logic issue rather than "
              f"pure speed being too high (see the per-iteration failure locations above).", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
