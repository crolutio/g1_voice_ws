#!/usr/bin/env bash
# Terminal 2: start the voice assistant: listening, conversation, gestures and walking in one.
# Makes sure Ollama (the local fallback model) is running, then launches the ROS 2 nodes.
# Groq is used when the GROQ_API_KEY environment variable is set.
# Extra arguments go to ros2 launch, e.g. use_microphone:=false to type instead of speaking.
set -eo pipefail
WS="$HOME/g1_voice_ws"
OLLAMA="$HOME/.local/ollama/bin/ollama"

if ! curl -sf http://localhost:11434/api/version >/dev/null; then
  echo "Starting Ollama (log: $WS/log/ollama.log)..."
  mkdir -p "$WS/log"
  nohup "$OLLAMA" serve >"$WS/log/ollama.log" 2>&1 &
  for _ in $(seq 30); do curl -sf http://localhost:11434/api/version >/dev/null && break; sleep 1; done
fi

source /opt/ros/jazzy/setup.bash
source "$WS/.venv/bin/activate"
source "$WS/install/setup.bash"
exec ros2 launch robot_voice_assistant voice_assistant.launch.py "$@"
