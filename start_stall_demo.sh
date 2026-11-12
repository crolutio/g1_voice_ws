#!/usr/bin/env bash
# The live stall demo: the GITEX hall with the camera on the robot's stall, the voice assistant listening on the
# microphone, and visitor after visitor walking up with something in their hand (training/live_stall.py).
# The robot greets each visitor and listens without the wake word: ask it what they hold, what it sees, anything
# else, or to wave, walk or turn (it keeps about 1 m from people). Say goodbye and the next visitor comes.
# Ctrl+C stops everything. Logs: log/stall_demo_scene.log, log/stall_demo_assistant.log.
set -eo pipefail
WS="$HOME/g1_voice_ws"
mkdir -p "$WS/log"
cd "$WS"

if [ -z "${GROQ_API_KEY:-}" ]; then
  echo "GROQ_API_KEY is not set: run this from a terminal where it is (see ~/.bashrc)." >&2
  exit 1
fi

pids=()
stop_all() {
  echo "Stopping the demo..."
  kill "${pids[@]}" 2>/dev/null || true
  pkill -f "ros2 launch robot_voice_assistant voice_assistant.launch.py" 2>/dev/null || true
  pkill -f "python -u walk_scene.py" 2>/dev/null || true
}
trap stop_all EXIT

# The local llama model is not used beside the simulation (config/stall_demo.yaml): free its GPU memory.
curl -s --max-time 10 http://localhost:11434/api/generate -d '{"model": "llama3.1:8b", "keep_alive": 0}' >/dev/null || true

echo "Starting the GITEX hall (log: log/stall_demo_scene.log)..."
bash start_isaac.sh --gitex --view stall > log/stall_demo_scene.log 2>&1 &
pids+=($!)
source /opt/ros/jazzy/setup.bash
until timeout 5 ros2 topic echo --once /gitex/visit_state 2>/dev/null | grep -q data; do sleep 3; done

echo "Starting the voice assistant with the microphone (log: log/stall_demo_assistant.log)..."
bash start_assistant.sh use_microphone:=true overrides:="$WS/src/robot_voice_assistant/config/stall_demo.yaml" \
  > log/stall_demo_assistant.log 2>&1 &
pids+=($!)

source "$WS/.venv/bin/activate"
python3 -u training/live_stall.py "$@"
