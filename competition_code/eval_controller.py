"""
eval_controller.py

eval_params(param_vector) -> float: runs a full evaluate_solution() race with the
given tunable_solution.py parameter vector loaded in, and returns a score to
minimize (lap time in seconds, or a penalty if it didn't finish).

Connects to CARLA once (lazily, on first call) and reuses that connection across
repeated eval_params() calls, since reconnecting per-trial would waste real time
in an optimization loop that may call this hundreds of times.
"""
import asyncio
import contextlib
import io
from typing import Optional

import carla
import numpy as np
import roar_py_carla

from competition_runner import evaluate_solution
from tunable_solution import TunableRoarCompetitionSolution, DEFAULT_PARAMS

FAILURE_PENALTY = 200.0  # seconds added on top of max_seconds when a run times out/fails to finish

_world = None
_roar_py_instance = None


def _ensure_connected():
    global _world, _roar_py_instance
    if _world is None:
        carla_client = carla.Client('127.0.0.1', 2000)
        carla_client.set_timeout(10.0)
        _roar_py_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)
        _world = _roar_py_instance.world
        _world.set_control_steps(0.05, 0.005)
        _world.set_asynchronous(False)
    return _world, _roar_py_instance


def _make_solution_class(params: np.ndarray):
    """evaluate_solution() calls solution_constructor(...) with fixed positional args
    only (no way to pass extra kwargs through), so bind this trial's params via a
    small subclass closure instead."""
    class _BoundTunableSolution(TunableRoarCompetitionSolution):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, params=params, **kwargs)
    return _BoundTunableSolution


async def eval_params_async(param_vector: np.ndarray, max_seconds: float = 300.0) -> float:
    world, roar_py_instance = _ensure_connected()
    solution_cls = _make_solution_class(np.array(param_vector, dtype=np.float64))

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = await evaluate_solution(
                world, solution_cls, max_seconds=max_seconds, enable_visualization=False
            )
    except Exception as e:
        # Bad params can produce NaN/inf control values, physics glitches, etc. -
        # don't let one broken trial kill the whole optimization run.
        print(f"eval_params: exception during run ({type(e).__name__}: {e}), treating as failure")
        roar_py_instance.clean_actors_not_registered()
        return max_seconds + FAILURE_PENALTY

    if result is None:
        return max_seconds + FAILURE_PENALTY
    return float(result["elapsed_time"])


def eval_params(param_vector: np.ndarray, max_seconds: float = 300.0) -> float:
    """Synchronous entry point - what the optimizer in Stage 3 will actually call."""
    return asyncio.run(eval_params_async(param_vector, max_seconds=max_seconds))


def close():
    global _world, _roar_py_instance
    if _roar_py_instance is not None:
        _roar_py_instance.close()
        _world = None
        _roar_py_instance = None


if __name__ == "__main__":
    # Quick sanity check: confirm the plumbing works end-to-end without running a
    # full baseline lap (which would take several minutes at flat 20 m/s x 3 laps).
    # A short max_seconds here should just hit the timeout path and return a penalty.
    print("Sanity check: running DEFAULT_PARAMS with a short max_seconds cap...")
    score = eval_params(DEFAULT_PARAMS, max_seconds=15.0)
    print(f"Score: {score:.1f} (expected ~{15.0 + FAILURE_PENALTY:.1f} penalty, since 15s isn't enough to finish 3 laps)")
    close()
