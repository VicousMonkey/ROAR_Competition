"""
record_manual_lap.py

Drive the vehicle manually (Xbox controller, falling back to keyboard) and log
every tick's waypoint index, location, velocity, heading, and applied control
to a CSV + .npz file. Used to produce raw driving data for build_speed_profile.py.

Does not modify submission.py / infrastructure.py.
"""
import argparse
import asyncio
import contextlib
import csv
import io
import re
import sys
import time
from typing import Optional, Dict, Any, List

import carla
import numpy as np
import pygame
import roar_py_carla
import roar_py_interface

from submission import filter_waypoints, normalize_rad
from competition_runner import RoarCompetitionRule

STEER_DEADZONE = 0.1
DEFAULT_STEER_SENSITIVITY = 0.25 # scales max steer output; 1.0 = full raw stick range
DEFAULT_STEER_EXPO = 3.0         # >1 softens response near center, full deflection still reaches max
MAX_STEER_DELTA_PER_TICK = 0.05     # ramp limit so a fast stick flick doesn't snap the wheel instantly
MAX_THROTTLE_DELTA_PER_TICK = 0.08  # ramp limit so trigger snaps don't instantly wheelspin
MAX_BRAKE_DELTA_PER_TICK = 0.12     # ramp limit so trigger snaps don't instantly lock the wheels
MOVE_START_SPEED_MPS = 0.5          # lap timer starts once speed exceeds this after a (re)spawn

# Default SDL/XInput axis guesses for an Xbox controller on Windows.
# Confirm these against the printed raw axis values on first run and
# override with --axis-steer / --axis-throttle / --axis-brake if different.
DEFAULT_AXIS_STEER = 0      # left stick X
DEFAULT_AXIS_THROTTLE = 5   # right trigger (RT)
DEFAULT_AXIS_BRAKE = 4      # left trigger (LT)
DEFAULT_BUTTON_RESPAWN = 0  # A button (standard SDL2 GameController mapping)

MINIMAP_SIZE = 190
MINIMAP_PAD = 14
HUD_MARGIN = 14
PANEL_BG = (15, 15, 20, 175)
PANEL_BORDER = (255, 255, 255, 55)

DEFAULT_AUTOPILOT_SPEED = 60.0 / 3.6  # m/s target speed while following route.txt (60 km/h)
_ROUTE_LINE_RE = re.compile(r"waypoints\s+(\d+)\s+at\s+\[([^\]]+)\]")


def trigger_to_unit(raw_value: float) -> float:
    """SDL reports triggers roughly in [-1, 1] with -1 = unpressed. Remap to [0, 1]."""
    return float(np.clip((raw_value + 1.0) / 2.0, 0.0, 1.0))


def load_route_waypoints(path: str) -> List[roar_py_interface.RoarPyWaypoint]:
    """Parse a route.txt log of 'reach waypoints IDX at [x y z]' lines into an
    ordered, de-duplicated list of RoarPyWaypoint (location only; heading/lane
    width aren't needed since the follower only reads .location)."""
    points_by_idx: Dict[int, np.ndarray] = {}
    order: List[int] = []
    with open(path) as f:
        for line in f:
            match = _ROUTE_LINE_RE.search(line)
            if not match:
                continue
            idx = int(match.group(1))
            coords = np.array([float(x) for x in match.group(2).split()])
            if idx not in points_by_idx:
                points_by_idx[idx] = coords
                order.append(idx)
    return [
        roar_py_interface.RoarPyWaypoint(points_by_idx[idx], np.zeros(3), 8.0)
        for idx in order
    ]


MIN_AUTOPILOT_SPEED = 3.0  # m/s floor, in case a waypoint's recorded speed was near-zero (e.g. dwelling at the start)


def load_route_from_recording(npz_path: str):
    """Build a route + per-waypoint target speed from a manual_lap_*.npz recording
    saved by this script. Uses the max recorded speed at each waypoint (not the
    average) so ticks spent stationary/dwelling don't drag the target speed down."""
    data = np.load(npz_path)
    waypoint_idx = data["waypoint_idx"]
    locations = np.stack([data["loc_x"], data["loc_y"], data["loc_z"]], axis=1)
    speeds = data["speed"]

    order: List[int] = []
    loc_by_idx: Dict[int, np.ndarray] = {}
    max_speed_by_idx: Dict[int, float] = {}
    for i in range(len(waypoint_idx)):
        idx = int(waypoint_idx[i])
        if idx not in loc_by_idx:
            loc_by_idx[idx] = locations[i]
            order.append(idx)
        max_speed_by_idx[idx] = max(max_speed_by_idx.get(idx, 0.0), float(speeds[i]))

    route_waypoints = [
        roar_py_interface.RoarPyWaypoint(loc_by_idx[idx], np.zeros(3), 8.0) for idx in order
    ]
    route_speeds = np.array([max(max_speed_by_idx[idx], MIN_AUTOPILOT_SPEED) for idx in order])
    return route_waypoints, route_speeds


class RecordingControlViewer:
    """Manual control viewer with gamepad (preferred) + keyboard fallback."""

    def __init__(self, axis_steer: int, axis_throttle: int, axis_brake: int, print_raw_axes: bool,
                 steer_sensitivity: float = DEFAULT_STEER_SENSITIVITY, steer_expo: float = DEFAULT_STEER_EXPO,
                 button_respawn: int = DEFAULT_BUTTON_RESPAWN,
                 track_waypoints: Optional[List[roar_py_interface.RoarPyWaypoint]] = None,
                 route_waypoints: Optional[List[roar_py_interface.RoarPyWaypoint]] = None,
                 route_speeds: Optional[np.ndarray] = None,
                 autopilot_speed: float = DEFAULT_AUTOPILOT_SPEED):
        self.screen = None
        self.clock = None
        self.joystick: Optional[pygame.joystick.JoystickType] = None
        self.axis_steer = axis_steer
        self.axis_throttle = axis_throttle
        self.axis_brake = axis_brake
        self.button_respawn = button_respawn
        self.print_raw_axes = print_raw_axes
        self.steer_sensitivity = steer_sensitivity
        self.steer_expo = steer_expo
        self._last_axis_print = 0.0
        self._applied_throttle = 0.0
        self._applied_brake = 0.0
        self._applied_steer = 0.0
        self.font_value = None
        self.font_label = None
        self.respawn_requested = False
        self.is_fullscreen = False
        self._windowed_size = None

        self.route_waypoints = route_waypoints
        self.route_speeds = route_speeds
        self.autopilot_speed = autopilot_speed
        self.autopilot_active = False
        self._autopilot_idx = 0

        self._minimap_track_points = []
        self._minimap_min_xy = np.zeros(2)
        self._minimap_scale = 1.0
        self._minimap_offset = np.zeros(2)
        if track_waypoints:
            self._build_minimap(track_waypoints)

    def _build_minimap(self, track_waypoints: List[roar_py_interface.RoarPyWaypoint]) -> None:
        locations = np.array([wp.location[:2] for wp in track_waypoints])
        min_xy = locations.min(axis=0)
        max_xy = locations.max(axis=0)
        span = np.maximum(max_xy - min_xy, 1e-3)
        inner = MINIMAP_SIZE - 2 * MINIMAP_PAD
        scale = inner / float(np.max(span))
        offset = (inner - span * scale) / 2.0 + MINIMAP_PAD

        self._minimap_min_xy = min_xy
        self._minimap_scale = scale
        self._minimap_offset = offset
        self._minimap_track_points = [
            (float((x - min_xy[0]) * scale + offset[0]), float((y - min_xy[1]) * scale + offset[1]))
            for x, y in locations
        ]

    def _minimap_transform(self, x: float, y: float):
        px = (x - self._minimap_min_xy[0]) * self._minimap_scale + self._minimap_offset[0]
        py = (y - self._minimap_min_xy[1]) * self._minimap_scale + self._minimap_offset[1]
        return px, py

    def init_pygame(self, x, y) -> None:
        pygame.init()
        pygame.font.init()
        self.font_value = pygame.font.SysFont("consolas", 30, bold=True)
        self.font_label = pygame.font.SysFont("consolas", 15)
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Gamepad detected: {self.joystick.get_name()} "
                  f"({self.joystick.get_numaxes()} axes, {self.joystick.get_numbuttons()} buttons)")
        else:
            print("No gamepad detected, falling back to keyboard controls (arrow keys).")

        self._windowed_size = (x, y)
        self.screen = pygame.display.set_mode((x, y), pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE)
        pygame.display.set_caption("RoarPy Manual Recording Viewer  (F11: toggle fullscreen)")
        pygame.key.set_repeat()
        self.clock = pygame.time.Clock()

    def toggle_fullscreen(self) -> None:
        if self.is_fullscreen:
            self.screen = pygame.display.set_mode(self._windowed_size, pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE)
            self.is_fullscreen = False
        else:
            self._windowed_size = self.screen.get_size()
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.is_fullscreen = True

    def close(self) -> None:
        pygame.quit()

    def _toggle_autopilot(self) -> None:
        if not self.route_waypoints:
            print("No route loaded (missing/empty route file), can't engage autopilot.")
            return
        self.autopilot_active = not self.autopilot_active
        print(f"Autopilot {'ENGAGED' if self.autopilot_active else 'disengaged'}"
              f"{' - following recorded route' if self.autopilot_active else ''}")

    def _draw_stat_panel(self, label: str, value: str, anchor: str, accent=(140, 210, 255)) -> None:
        label_surf = self.font_label.render(label, True, (190, 195, 205))
        value_surf = self.font_value.render(value, True, accent)

        pad_x, pad_y, gap = 16, 8, 2
        w = max(label_surf.get_width(), value_surf.get_width()) + pad_x * 2
        h = label_surf.get_height() + value_surf.get_height() + pad_y * 2 + gap

        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        rect = panel.get_rect()
        pygame.draw.rect(panel, PANEL_BG, rect, border_radius=10)
        pygame.draw.rect(panel, PANEL_BORDER, rect, width=1, border_radius=10)
        panel.blit(label_surf, ((w - label_surf.get_width()) // 2, pad_y))
        panel.blit(value_surf, ((w - value_surf.get_width()) // 2, pad_y + label_surf.get_height() + gap))

        screen_w, screen_h = self.screen.get_size()
        if anchor == "topmid":
            pos = ((screen_w - w) // 2, HUD_MARGIN)
        elif anchor == "bottomright":
            pos = (screen_w - w - HUD_MARGIN, screen_h - h - HUD_MARGIN)
        elif anchor == "bottomleft":
            pos = (HUD_MARGIN, screen_h - h - HUD_MARGIN)
        elif anchor == "topleft2":
            pos = (HUD_MARGIN, HUD_MARGIN + self.font_label.get_height() + 10)
        else:
            pos = (HUD_MARGIN, HUD_MARGIN)
        self.screen.blit(panel, pos)

    def _draw_minimap(self, vehicle_location: Optional[np.ndarray]) -> None:
        panel = pygame.Surface((MINIMAP_SIZE, MINIMAP_SIZE), pygame.SRCALPHA)
        rect = panel.get_rect()
        pygame.draw.rect(panel, PANEL_BG, rect, border_radius=10)
        pygame.draw.rect(panel, PANEL_BORDER, rect, width=1, border_radius=10)

        if len(self._minimap_track_points) >= 2:
            pygame.draw.lines(panel, (120, 200, 255, 230), True, self._minimap_track_points, 2)
            start_x, start_y = self._minimap_track_points[0]
            pygame.draw.circle(panel, (90, 255, 130), (int(start_x), int(start_y)), 4)

        if vehicle_location is not None and len(self._minimap_track_points) >= 2:
            car_x, car_y = self._minimap_transform(vehicle_location[0], vehicle_location[1])
            pygame.draw.circle(panel, (255, 70, 70), (int(car_x), int(car_y)), 5)
            pygame.draw.circle(panel, (255, 255, 255), (int(car_x), int(car_y)), 5, width=1)

        screen_w, screen_h = self.screen.get_size()
        self.screen.blit(panel, (HUD_MARGIN, screen_h - MINIMAP_SIZE - HUD_MARGIN))

    def _read_gamepad(self) -> Dict[str, Any]:
        control = {"throttle": 0.0, "steer": 0.0, "brake": 0.0}

        num_axes = self.joystick.get_numaxes()
        raw_axes = [self.joystick.get_axis(i) for i in range(num_axes)]

        if self.print_raw_axes:
            now = time.time()
            if now - self._last_axis_print > 0.5:
                self._last_axis_print = now
                pressed_buttons = [i for i in range(self.joystick.get_numbuttons()) if self.joystick.get_button(i)]
                print("raw axes:", [f"{v:+.2f}" for v in raw_axes], "pressed buttons:", pressed_buttons)

        if self.axis_steer < num_axes:
            steer_raw = raw_axes[self.axis_steer]
            if abs(steer_raw) < STEER_DEADZONE:
                steer_raw = 0.0
            else:
                # rescale so the deadzone edge maps to 0 and full deflection still maps to 1
                sign = np.sign(steer_raw)
                steer_raw = sign * (abs(steer_raw) - STEER_DEADZONE) / (1.0 - STEER_DEADZONE)
            # expo curve softens response near center while still reaching full lock at full deflection
            steer_shaped = np.sign(steer_raw) * (abs(steer_raw) ** self.steer_expo)
            control["steer"] = float(np.clip(steer_shaped * self.steer_sensitivity, -1.0, 1.0))

        if self.axis_throttle < num_axes:
            control["throttle"] = trigger_to_unit(raw_axes[self.axis_throttle])

        if self.axis_brake < num_axes:
            control["brake"] = trigger_to_unit(raw_axes[self.axis_brake])

        return control

    def _read_keyboard(self) -> Dict[str, Any]:
        control = {"throttle": 0.0, "steer": 0.0, "brake": 0.0}
        pressed_keys = pygame.key.get_pressed()
        if pressed_keys[pygame.K_UP]:
            control['throttle'] = 0.4
        if pressed_keys[pygame.K_DOWN]:
            control['brake'] = 0.2
        if pressed_keys[pygame.K_LEFT]:
            control['steer'] = -0.2
        if pressed_keys[pygame.K_RIGHT]:
            control['steer'] = 0.2
        return control

    def _autopilot_control(
        self, vehicle_location: np.ndarray, vehicle_rotation: np.ndarray, speed_mps: float
    ) -> Dict[str, Any]:
        """Follow self.route_waypoints with the same simple pursuit + speed
        P-controller shape as submission.py's baseline solution."""
        self._autopilot_idx = filter_waypoints(vehicle_location, self._autopilot_idx, self.route_waypoints)
        target_idx = (self._autopilot_idx + 3) % len(self.route_waypoints)
        target = self.route_waypoints[target_idx]

        vector_to_target = (target.location - vehicle_location)[:2]
        heading_to_target = np.arctan2(vector_to_target[1], vector_to_target[0])
        delta_heading = normalize_rad(heading_to_target - vehicle_rotation[2])

        steer = (
            -8.0 / np.sqrt(speed_mps) * delta_heading / np.pi
        ) if speed_mps > 1e-2 else -np.sign(delta_heading)
        steer = float(np.clip(steer, -1.0, 1.0))

        target_speed = (
            float(self.route_speeds[target_idx]) if self.route_speeds is not None else self.autopilot_speed
        )
        throttle_raw = 0.05 * (target_speed - speed_mps)
        return {
            "throttle": float(np.clip(throttle_raw, 0.0, 1.0)),
            "steer": steer,
            "brake": float(np.clip(-throttle_raw, 0.0, 1.0)),
        }

    def render(
        self,
        image: roar_py_interface.RoarPyCameraSensorData,
        speed_mps: float = 0.0,
        lap_time_seconds: float = 0.0,
        vehicle_location: Optional[np.ndarray] = None,
        vehicle_rotation: Optional[np.ndarray] = None,
        recording_enabled: bool = True,
    ) -> Optional[Dict[str, Any]]:
        image_pil = image.get_image()
        if self.screen is None:
            self.init_pygame(image_pil.width, image_pil.height)

        self.respawn_requested = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                self.respawn_requested = True
            if event.type == pygame.JOYBUTTONDOWN and event.button == self.button_respawn:
                self.respawn_requested = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                self.toggle_fullscreen()
            if event.type == pygame.VIDEORESIZE and not self.is_fullscreen:
                self._windowed_size = (event.w, event.h)
                self.screen = pygame.display.set_mode(
                    (event.w, event.h), pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE
                )
            # D-pad is reported as a "hat" on standard SDL2 gamepad mapping; (0, -1) = down
            if event.type == pygame.JOYHATMOTION and event.value[1] == -1:
                self._toggle_autopilot()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self._toggle_autopilot()

        if self.autopilot_active and self.route_waypoints and vehicle_location is not None and vehicle_rotation is not None:
            new_control = self._autopilot_control(vehicle_location, vehicle_rotation, speed_mps)
        else:
            new_control = self._read_gamepad() if self.joystick is not None else self._read_keyboard()

        # Ramp steer/throttle/brake instead of snapping, so a sudden stick/trigger
        # movement doesn't instantly break wheel traction or feel oversensitive.
        self._applied_steer += float(np.clip(
            new_control["steer"] - self._applied_steer,
            -MAX_STEER_DELTA_PER_TICK, MAX_STEER_DELTA_PER_TICK
        ))
        new_control["steer"] = self._applied_steer

        self._applied_throttle += float(np.clip(
            new_control["throttle"] - self._applied_throttle,
            -MAX_THROTTLE_DELTA_PER_TICK, MAX_THROTTLE_DELTA_PER_TICK
        ))
        self._applied_brake += float(np.clip(
            new_control["brake"] - self._applied_brake,
            -MAX_BRAKE_DELTA_PER_TICK, MAX_BRAKE_DELTA_PER_TICK
        ))
        new_control["throttle"] = self._applied_throttle
        new_control["brake"] = self._applied_brake

        new_control["hand_brake"] = np.array([0])
        new_control["reverse"] = np.array([0])

        image_surface = pygame.image.fromstring(image_pil.tobytes(), image_pil.size, image_pil.mode).convert()
        screen_w, screen_h = self.screen.get_size()
        if (screen_w, screen_h) != image_surface.get_size():
            image_surface = pygame.transform.smoothscale(image_surface, (screen_w, screen_h))
        self.screen.fill((0, 0, 0))
        self.screen.blit(image_surface, (0, 0))

        speed_mph = speed_mps * 2.23694
        self._draw_stat_panel("LAP TIME", f"{lap_time_seconds:.2f} s", anchor="topmid", accent=(140, 210, 255))
        self._draw_stat_panel("SPEED", f"{speed_mph:.1f} mph", anchor="bottomright", accent=(255, 205, 90))
        self._draw_minimap(vehicle_location)

        hint_text = "A: respawn at start   |   D-pad Down: toggle autopilot"
        hint_surf = self.font_label.render(hint_text, True, (200, 200, 200))
        self.screen.blit(hint_surf, (HUD_MARGIN, HUD_MARGIN))

        if self.autopilot_active:
            self._draw_stat_panel("MODE", "AUTOPILOT", anchor="topleft2", accent=(120, 255, 140))
        elif recording_enabled:
            self._draw_stat_panel("", "● REC", anchor="topleft2", accent=(255, 90, 90))
        else:
            self._draw_stat_panel("", "WARMUP (not recording)", anchor="topleft2", accent=(200, 200, 200))

        pygame.display.flip()
        self.clock.tick(60)
        return new_control


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--laps", type=int, default=1, help="Number of laps to record")
    parser.add_argument("--out-prefix", type=str, default="manual_lap", help="Output file prefix")
    parser.add_argument("--axis-steer", type=int, default=DEFAULT_AXIS_STEER)
    parser.add_argument("--axis-throttle", type=int, default=DEFAULT_AXIS_THROTTLE)
    parser.add_argument("--axis-brake", type=int, default=DEFAULT_AXIS_BRAKE)
    parser.add_argument("--button-respawn", type=int, default=DEFAULT_BUTTON_RESPAWN,
                         help="Joystick button index that triggers respawn-at-start (default 0 = A on a standard Xbox controller)")
    parser.add_argument("--print-raw-axes", action="store_true",
                         help="Continuously print raw joystick axis values to help identify axis mapping")
    parser.add_argument("--steer-sensitivity", type=float, default=DEFAULT_STEER_SENSITIVITY,
                         help="Scales max steer output (1.0 = full raw stick range, lower = less sensitive)")
    parser.add_argument("--steer-expo", type=float, default=DEFAULT_STEER_EXPO,
                         help="Expo curve exponent; >1 softens response near center, still reaches full lock at full deflection")
    parser.add_argument("--route-file", type=str, default="route.txt",
                         help="Plain waypoint-only route log to follow in autopilot mode (used if --route-npz isn't given)")
    parser.add_argument("--route-npz", type=str, default=None,
                         help="A manual_lap_*.npz recording to follow in autopilot mode, replaying your recorded speed "
                              "at each waypoint instead of a flat --autopilot-speed")
    parser.add_argument("--autopilot-speed", type=float, default=DEFAULT_AUTOPILOT_SPEED,
                         help="Flat target speed (m/s) while following --route-file (ignored if --route-npz is used)")
    args = parser.parse_args()

    route_waypoints = None
    route_speeds = None
    if args.route_npz:
        try:
            route_waypoints, route_speeds = load_route_from_recording(args.route_npz)
            print(f"Loaded {len(route_waypoints)} route points with recorded speed from {args.route_npz}")
        except FileNotFoundError:
            print(f"Route recording {args.route_npz} not found, autopilot will be unavailable.")
    else:
        try:
            route_waypoints = load_route_waypoints(args.route_file)
            if route_waypoints:
                print(f"Loaded {len(route_waypoints)} route points from {args.route_file} "
                      f"(flat {args.autopilot_speed * 3.6:.0f} km/h target, no recorded speed)")
            else:
                print(f"No route points parsed from {args.route_file}, autopilot will be unavailable.")
                route_waypoints = None
        except FileNotFoundError:
            print(f"Route file {args.route_file} not found, autopilot will be unavailable.")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    csv_path = f"{args.out_prefix}_{run_id}.csv"
    npz_path = f"{args.out_prefix}_{run_id}.npz"

    carla_client = carla.Client('127.0.0.1', 2000)
    carla_client.set_timeout(5.0)
    roar_py_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)
    world = roar_py_instance.world
    world.set_control_steps(0.05, 0.005)
    world.set_asynchronous(False)

    waypoints = world.maneuverable_waypoints
    vehicle = world.spawn_vehicle(
        "vehicle.audi.tt",
        waypoints[0].location + np.array([0, 0, 1]),
        waypoints[0].roll_pitch_yaw,
        True,
    )
    camera = vehicle.attach_camera_sensor(
        roar_py_interface.RoarPyCameraSensorDataRGB,
        np.array([-2.0 * vehicle.bounding_box.extent[0], 0.0, 3.0 * vehicle.bounding_box.extent[2]]),
        np.array([0, 10 / 180.0 * np.pi, 0]),
        image_width=1024,
        image_height=768
    )
    location_sensor = vehicle.attach_location_in_world_sensor()
    velocity_sensor = vehicle.attach_velocimeter_sensor()
    rpy_sensor = vehicle.attach_roll_pitch_yaw_sensor()
    collision_sensor = vehicle.attach_collision_sensor(np.zeros(3), np.zeros(3))

    rule = RoarCompetitionRule(waypoints * max(args.laps, 1), vehicle, world)

    for _ in range(20):
        await world.step()
    rule.initialize_race()

    await vehicle.receive_observation()

    viewer = RecordingControlViewer(
        args.axis_steer, args.axis_throttle, args.axis_brake, args.print_raw_axes,
        steer_sensitivity=args.steer_sensitivity, steer_expo=args.steer_expo,
        button_respawn=args.button_respawn, track_waypoints=waypoints,
        route_waypoints=route_waypoints, route_speeds=route_speeds, autopilot_speed=args.autopilot_speed
    )

    current_waypoint_idx = 10
    vehicle_location = location_sensor.get_last_gym_observation()
    current_waypoint_idx = filter_waypoints(vehicle_location, current_waypoint_idx, waypoints)

    log_rows = []
    lap_timer_running = False
    lap_start_wall_time = None
    recording_enabled = False

    print(f"Recording up to {args.laps} lap(s). Close the viewer window to stop and save early.")
    print("Press A at any time to respawn at the start and restart the lap.")
    print("Nothing is logged until you respawn/restart once (A, or an auto-respawn after a collision) - "
          "drive a warmup pass, then hit A to start the recorded attempt.")

    try:
        tick = 0
        while True:
            await vehicle.receive_observation()
            with contextlib.redirect_stdout(io.StringIO()):
                await rule.tick()  # rule.tick() prints a "reach waypoints ..." debug line every tick; silence it

            vehicle_location = location_sensor.get_last_gym_observation()
            vehicle_rotation = rpy_sensor.get_last_gym_observation()
            vehicle_velocity = velocity_sensor.get_last_gym_observation()
            speed = float(np.linalg.norm(vehicle_velocity))

            current_waypoint_idx = filter_waypoints(vehicle_location, current_waypoint_idx, waypoints)

            collision_impulse_norm = np.linalg.norm(collision_sensor.get_last_observation().impulse_normal)
            if collision_impulse_norm > 100.0:
                print(f"major collision (impulse {collision_impulse_norm:.1f}), respawning")
                await rule.respawn()
                lap_timer_running = False
                lap_start_wall_time = None
                log_rows = []
                recording_enabled = True
                await vehicle.receive_observation()
                vehicle_location = location_sensor.get_last_gym_observation()
                current_waypoint_idx = filter_waypoints(vehicle_location, current_waypoint_idx, waypoints)
                continue

            if rule.lap_finished():
                print("Lap target reached.")
                break

            if not lap_timer_running and speed > MOVE_START_SPEED_MPS:
                lap_timer_running = True
                lap_start_wall_time = time.time()
            lap_time_seconds = (time.time() - lap_start_wall_time) if lap_timer_running else 0.0

            control = viewer.render(
                camera.get_last_observation(), speed_mps=speed, lap_time_seconds=lap_time_seconds,
                vehicle_location=vehicle_location, vehicle_rotation=vehicle_rotation,
                recording_enabled=recording_enabled
            )
            if control is None:
                print("Viewer closed, stopping early.")
                break

            if viewer.respawn_requested:
                print("Respawn requested (A pressed), resetting to start of lap.")
                await rule.respawn()
                lap_timer_running = False
                lap_start_wall_time = None
                log_rows = []
                was_recording = recording_enabled
                recording_enabled = True
                if not was_recording:
                    print("Recording is now ON for this attempt.")
                await vehicle.receive_observation()
                vehicle_location = location_sensor.get_last_gym_observation()
                current_waypoint_idx = filter_waypoints(vehicle_location, current_waypoint_idx, waypoints)
                continue

            if recording_enabled:
                log_rows.append({
                    "tick": tick,
                    "waypoint_idx": current_waypoint_idx % len(waypoints),
                    "loc_x": vehicle_location[0],
                    "loc_y": vehicle_location[1],
                    "loc_z": vehicle_location[2],
                    "vel_x": vehicle_velocity[0],
                    "vel_y": vehicle_velocity[1],
                    "vel_z": vehicle_velocity[2],
                    "speed": speed,
                    "roll": vehicle_rotation[0],
                    "pitch": vehicle_rotation[1],
                    "yaw": vehicle_rotation[2],
                    "throttle": control["throttle"],
                    "steer": control["steer"],
                    "brake": control["brake"],
                })

            await vehicle.apply_action(control)
            await world.step()
            tick += 1
    finally:
        vehicle.close()
        viewer.close()
        roar_py_instance.close()

        if log_rows:
            fieldnames = list(log_rows[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(log_rows)

            arrays = {key: np.array([row[key] for row in log_rows]) for key in fieldnames}
            arrays["run_id"] = np.array(run_id)
            arrays["num_waypoints"] = np.array(len(waypoints))
            np.savez_compressed(npz_path, **arrays)

            print(f"Saved {len(log_rows)} ticks to {csv_path} and {npz_path}")
        else:
            print("No ticks recorded, nothing saved.")

        print("Done, exiting.")


if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)
