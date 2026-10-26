"""Train (or play back) the R1 walking policy with Isaac Lab, after registering the custom task.

    ~/g1_voice_ws/training/train_walk.sh                      # full training
    ~/g1_voice_ws/training/train_walk.sh --max_iterations 10  # quick test
    ~/g1_voice_ws/training/train_walk.sh --gpu_duty 0.5 ...   # GPU busy only ~50% of the time (slower)
"""

import os
import sys
import time

import warp as wp

wp.config.enable_backward = False

sys.path.insert(0, os.path.dirname(__file__))
import walk_task  # noqa: E402,F401  (registers R1-Walk-Flat)
from isaaclab_rl.entrypoints import run_play_cli, run_train_cli  # noqa: E402


def _take_gpu_duty(argv: list[str]) -> tuple[float, list[str]]:
    """Remove '--gpu_duty X' from argv; return X (1.0 = no throttling)."""
    if "--gpu_duty" not in argv:
        return 1.0, argv
    index = argv.index("--gpu_duty")
    duty = float(argv[index + 1])
    if not 0.05 <= duty <= 1.0:
        raise SystemExit("--gpu_duty must be between 0.05 and 1.0")
    return duty, argv[:index] + argv[index + 2 :]


def throttle_environment_steps(duty: float) -> None:
    """After every simulation step, pause so the GPU is busy only `duty` of the time."""
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    original_step = RslRlVecEnvWrapper.step

    def throttled_step(self, actions):
        started = time.monotonic()
        result = original_step(self, actions)
        import torch

        torch.cuda.synchronize()  # count the GPU work of this step, not just its launch
        time.sleep((time.monotonic() - started) * (1.0 / duty - 1.0))
        return result

    RslRlVecEnvWrapper.step = throttled_step
    print(f"[train_walk] GPU duty cycle {duty:.0%}: pausing after each simulation step.", flush=True)


if __name__ == "__main__":
    gpu_duty, arguments = _take_gpu_duty(sys.argv[1:])
    if gpu_duty < 1.0:
        throttle_environment_steps(gpu_duty)
    if arguments and arguments[0] == "play":
        raise SystemExit(run_play_cli(arguments[1:]))
    raise SystemExit(run_train_cli(arguments))
