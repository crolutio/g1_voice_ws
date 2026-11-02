#!/usr/bin/env bash
# Keep an Isaac Sim window on the newest checkpoint of a training run while it trains, reloading every STEP
# iterations, and leave it on the final checkpoint when training ends.
#   AFTER=<run dir name> STEP=100 bash watch_training.sh   # only runs newer than AFTER
# Env: EXPERIMENT (log folder, default r1_walk_plan), AFTER, PLAY_TASK, STEP (default 200), NUM_ENVS (default 32).
set -u
source "$HOME/isaacsim-env/bin/activate"
cd "$HOME/g1_voice_ws/training"
ROOT="logs/rsl_rl/${EXPERIMENT:-r1_walk_plan}"
STEP="${STEP:-200}"
shown=-100000
play_pid=""
log() { echo "$(date +%H:%M) $*"; }

newest_run() {  # the newest run that already has a checkpoint
  for run in $(ls -1 "$ROOT" | sort -r | awk -v after="${AFTER:-}" '$0 > after'); do
    ls "$ROOT/$run"/model_*.pt > /dev/null 2>&1 && { echo "$run"; return; }
  done
}
newest_model() { ls -1 "$1"/model_*.pt 2>/dev/null | sed 's/.*model_\([0-9]*\)\.pt/\1/' | sort -n | tail -1; }
training_running() { pgrep -f "train_walk[.]py --rl_library" > /dev/null; }

while true; do
  run="$(newest_run)"
  if [ -n "$run" ]; then
    latest="$(newest_model "$ROOT/$run")"
    if [ -n "$latest" ] && { [ $((latest - shown)) -ge "$STEP" ] || { ! training_running && [ "$latest" != "$shown" ]; }; }; then
      [ -n "$play_pid" ] && kill "$play_pid" 2>/dev/null && wait "$play_pid" 2>/dev/null
      log "showing $run/model_$latest.pt"
      python -u train_walk.py play --rl_library rsl_rl --task "${PLAY_TASK:-R1-Walk-Plan-Flat-Play}" \
        --num_envs "${NUM_ENVS:-32}" --viz newton_gl --checkpoint "$PWD/$ROOT/$run/model_$latest.pt" \
        physics=isaacsim_physx > play_watch.log 2>&1 &
      play_pid=$!
      shown=$latest
    fi
  fi
  if ! training_running && [ "$shown" -ge 0 ]; then
    log "training finished; leaving the window on model_$shown.pt"
    exit 0
  fi
  sleep 30
done
