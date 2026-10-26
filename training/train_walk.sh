#!/usr/bin/env bash
# Train the R1 walking policy (PhysX, Unitree R1, legs only). Logs and checkpoints go to
# ~/g1_voice_ws/training/logs/rsl_rl/r1_walk_flat/. Extra arguments are passed on,
# e.g. --num_envs 2048 --max_iterations 1500, or --resume to continue the last run.
set -eo pipefail
source "$HOME/isaacsim-env/bin/activate"
cd "$HOME/g1_voice_ws/training"
exec python -u train_walk.py --rl_library rsl_rl --task R1-Walk-Flat "$@" physics=isaacsim_physx
