"""
optimize_controller.py

CMA-ES search over tunable_solution.py's 23 controller parameters, using
eval_controller.eval_params() (a full evaluate_solution() race) as the objective
to minimize.

Recommended over Bayesian optimization (e.g. Optuna) for this problem because:
  - 23 continuous dims is past where GP/TPE surrogate quality holds up well;
    CMA-ES natively scales into the tens-of-dimensions range.
  - CMA-ES's population-ranking updates are more robust to a noisy objective
    (real CARLA physics/timing variance) than a surrogate fit to raw values.
  - At the eval budget this problem realistically needs (a few hundred trials,
    since each trial costs real simulated-race minutes), BO's usual sample-
    efficiency advantage over CMA-ES mostly disappears.

Checkpoints after every generation (--checkpoint-file, a pickle of the live
CMAEvolutionStrategy) so a long run can be safely killed and resumed with
--resume. Logs every individual trial (--log-file, CSV) for progress tracking
and for Stage 4's "pull out the best params" step.
"""
import argparse
import csv
import os
import pickle
import time

import cma
import numpy as np

from eval_controller import eval_params, close as close_eval
from tunable_solution import DEFAULT_PARAMS, PARAM_BOUNDS, PARAM_NAMES


def load_or_init_es(checkpoint_file: str, resume: bool, popsize: int):
    if resume and os.path.exists(checkpoint_file):
        with open(checkpoint_file, "rb") as f:
            es = pickle.load(f)
        print(f"Resumed CMA-ES from {checkpoint_file} at generation {es.countiter}, "
              f"{es.countevals} evals so far.")
        return es

    lower_bounds = [b[0] for b in PARAM_BOUNDS]
    upper_bounds = [b[1] for b in PARAM_BOUNDS]
    # Per-parameter initial spread, since these params live on very different
    # scales (lookahead ~1-8 vs throttle_gain ~0.01-0.2 vs speeds ~5-35) - a
    # single scalar sigma0 would badly over/under-explore some dimensions.
    initial_stds = [(hi - lo) * 0.3 for lo, hi in PARAM_BOUNDS]

    options = {
        "bounds": [lower_bounds, upper_bounds],
        "CMA_stds": initial_stds,
        "popsize": popsize,
        "verbose": -3,
    }
    es = cma.CMAEvolutionStrategy(DEFAULT_PARAMS.tolist(), 1.0, options)
    print(f"Starting fresh CMA-ES run: {len(PARAM_NAMES)} params, popsize={es.popsize}")
    return es


def append_log(log_file: str, generation: int, trial_in_gen: int, params: np.ndarray, score: float, elapsed_s: float):
    write_header = not os.path.exists(log_file)
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["generation", "trial_in_gen", "score", "eval_seconds"] + PARAM_NAMES)
        writer.writerow([generation, trial_in_gen, score, f"{elapsed_s:.1f}"] + [f"{p:.6f}" for p in params])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-evals", type=int, default=150,
                         help="Stop after roughly this many total evaluations (rounded up to a full generation)")
    parser.add_argument("--max-seconds-per-eval", type=float, default=240.0,
                         help="Per-trial cap passed to eval_params(); bad params get cut off here instead of hanging")
    parser.add_argument("--popsize", type=int, default=None, help="Override CMA-ES population size (default: auto)")
    parser.add_argument("--log-file", type=str, default="optimize_controller_log.csv")
    parser.add_argument("--checkpoint-file", type=str, default="optimize_controller_checkpoint.pkl")
    parser.add_argument("--resume", action="store_true", help="Resume from --checkpoint-file if it exists")
    args = parser.parse_args()

    popsize = args.popsize or (4 + int(3 * np.log(len(PARAM_NAMES))))
    es = load_or_init_es(args.checkpoint_file, args.resume, popsize)

    evals_at_start = es.countevals
    best_score_seen = float("inf")

    try:
        while not es.stop() and (es.countevals - evals_at_start) < args.budget_evals:
            generation = es.countiter + 1
            candidates = es.ask()
            scores = []

            for i, candidate in enumerate(candidates):
                clipped = np.clip(candidate, [b[0] for b in PARAM_BOUNDS], [b[1] for b in PARAM_BOUNDS])
                t0 = time.time()
                score = eval_params(clipped, max_seconds=args.max_seconds_per_eval)
                elapsed = time.time() - t0
                scores.append(score)
                append_log(args.log_file, generation, i, clipped, score, elapsed)

                marker = " <- new best" if score < best_score_seen else ""
                best_score_seen = min(best_score_seen, score)
                print(f"[gen {generation}] trial {i+1}/{len(candidates)}: score={score:.1f}s "
                      f"({elapsed:.0f}s wall) | best so far={best_score_seen:.1f}s{marker}", flush=True)

            es.tell(candidates, scores)
            with open(args.checkpoint_file, "wb") as f:
                pickle.dump(es, f)
            print(f"=== Generation {generation} done, checkpoint saved. "
                  f"Total evals: {es.countevals}. Best so far: {best_score_seen:.1f}s ===\n", flush=True)

    finally:
        close_eval()

    print(f"Stopped. Total evals: {es.countevals}. Best score seen this run: {best_score_seen:.1f}s")
    print(f"Full trial log: {args.log_file}")
    print(f"Checkpoint (for --resume): {args.checkpoint_file}")


if __name__ == "__main__":
    main()
