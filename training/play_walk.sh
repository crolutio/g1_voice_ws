#!/usr/bin/env bash
# Watch 32 robots walk with random speed commands, using the newest trained checkpoint
# (or --checkpoint PATH). Also exports the policy to logs/.../exported/policy.pt and policy.onnx.
set -eo pipefail
source "$HOME/isaacsim-env/bin/activate"
cd "$HOME/g1_voice_ws/training"
exec python -u train_walk.py play --rl_library rsl_rl --task R1-Walk-Flat-Play --viz kit "$@" physics=isaacsim_physx
