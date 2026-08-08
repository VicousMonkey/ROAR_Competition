"""
tunable_solution.py

A parameterized copy of the baseline controller logic from submission.py, built
for offline black-box optimization. Does NOT modify submission.py.

Design choices:
- Section speed profile is segmented by EQUAL WAYPOINT-INDEX RANGES, not detected
  curvature. The track's waypoints are spaced ~2m apart fairly uniformly (confirmed
  from prior telemetry), so equal-index sections are a good proxy for equal-length
  sections. Curvature-based segmentation (estimating local heading change per
  waypoint to place more sections on tight corners) would be more "principled" but
  adds real geometry code and bug surface for uncertain benefit at this parameter
  budget - the optimizer can still discover a corner needs to be slow, since
  whichever section a corner falls into gets its own free speed parameter anyway.
- NUM_SPEED_SECTIONS = 20 keeps the total parameter count at 23 (1 lookahead +
  1 steer gain + 1 throttle gain + 20 section speeds), comfortably inside the
  15-40 budget with room to increase resolution later if needed.
"""
from typing import List, Optional
import numpy as np
import roar_py_interface

from submission import filter_waypoints, normalize_rad

NUM_SPEED_SECTIONS = 20

# Parameter vector layout: [lookahead, steer_gain, throttle_gain, section_speed_0, ..., section_speed_{N-1}]
PARAM_NAMES = ["lookahead", "steer_gain", "throttle_gain"] + [
    f"section_speed_{i}" for i in range(NUM_SPEED_SECTIONS)
]

# Matches the current baseline exactly: lookahead=3, steer gain=8.0, throttle gain=0.05,
# flat 20 m/s everywhere. Loading these defaults must reproduce today's behavior unchanged.
DEFAULT_PARAMS = np.array(
    [3.0, 8.0, 0.05] + [20.0] * NUM_SPEED_SECTIONS,
    dtype=np.float64,
)

# (low, high) bounds per parameter, same order as PARAM_NAMES.
PARAM_BOUNDS = (
    [(1.0, 8.0), (2.0, 20.0), (0.01, 0.2)]
    + [(5.0, 35.0)] * NUM_SPEED_SECTIONS
)


def section_index_for_waypoint(waypoint_idx: int, num_waypoints: int, num_sections: int = NUM_SPEED_SECTIONS) -> int:
    idx = int(waypoint_idx / num_waypoints * num_sections)
    return min(idx, num_sections - 1)


class TunableRoarCompetitionSolution:
    """Same constructor signature as submission.RoarCompetitionSolution so it can be
    passed directly as evaluate_solution()'s solution_constructor. Control logic is
    identical to the baseline except every tunable constant is read from self.params."""

    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle: roar_py_interface.RoarPyActor,
        camera_sensor: roar_py_interface.RoarPyCameraSensor = None,
        location_sensor: roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor: roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor: roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor: roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor: roar_py_interface.RoarPyCollisionSensor = None,
        params: Optional[np.ndarray] = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor
        self.params = np.array(params if params is not None else DEFAULT_PARAMS, dtype=np.float64)

    @property
    def lookahead(self) -> int:
        return int(round(self.params[0]))

    @property
    def steer_gain(self) -> float:
        return float(self.params[1])

    @property
    def throttle_gain(self) -> float:
        return float(self.params[2])

    def section_speed(self, waypoint_idx: int) -> float:
        section = section_index_for_waypoint(waypoint_idx, len(self.maneuverable_waypoints))
        return float(self.params[3 + section])

    async def initialize(self) -> None:
        vehicle_location = self.location_sensor.get_last_gym_observation()
        self.current_waypoint_idx = 10
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location, self.current_waypoint_idx, self.maneuverable_waypoints
        )

    async def step(self) -> None:
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_velocity_norm = np.linalg.norm(vehicle_velocity)

        self.current_waypoint_idx = filter_waypoints(
            vehicle_location, self.current_waypoint_idx, self.maneuverable_waypoints
        )
        waypoint_to_follow = self.maneuverable_waypoints[
            (self.current_waypoint_idx + self.lookahead) % len(self.maneuverable_waypoints)
        ]

        vector_to_waypoint = (waypoint_to_follow.location - vehicle_location)[:2]
        heading_to_waypoint = np.arctan2(vector_to_waypoint[1], vector_to_waypoint[0])
        delta_heading = normalize_rad(heading_to_waypoint - vehicle_rotation[2])

        steer_control = (
            -self.steer_gain / np.sqrt(vehicle_velocity_norm) * delta_heading / np.pi
        ) if vehicle_velocity_norm > 1e-2 else -np.sign(delta_heading)
        steer_control = np.clip(steer_control, -1.0, 1.0)

        target_speed = self.section_speed(self.current_waypoint_idx)
        throttle_control = self.throttle_gain * (target_speed - vehicle_velocity_norm)

        control = {
            "throttle": np.clip(throttle_control, 0.0, 1.0),
            "steer": steer_control,
            "brake": np.clip(-throttle_control, 0.0, 1.0),
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0,
        }
        await self.vehicle.apply_action(control)
        return control
