# R1 Voice Assistant

A simulated **Unitree R1 EDU** humanoid robot (26 degrees of freedom, no hands) that you can talk to and that walks when you ask it to. It runs in NVIDIA Isaac Sim and is connected through ROS 2.

- **Listens** to your microphone and turns speech into text (Whisper large-v3-turbo, on the GPU).
- **Thinks** of an answer with Groq's `qwen/qwen3.8-27b` (in the cloud), falling back to a local `llama3.1:8b` when Groq is unavailable.
- **Speaks** the answer out loud with an NVIDIA Riva voice (Magpie TTS in NVIDIA's cloud), or with Qwen3-TTS on this computer when there is no NVIDIA key or no internet.
- **Moves**: gestures with its arms, waist and head while it talks, and walks forward and backward and turns when asked.
- **Knows and remembers**: Qwen3-Embedding searches a folder of documents for each question, remembers earlier conversations, and recognises clear commands by meaning.

Conversation, gestures and walking all work together in one mode, started from **two terminals**. Say the **wake word** "robot" to start a conversation; from then on it answers everything without the wake word, until you say **"goodbye"** (or "stop listening", "that's all", "go to sleep").

## Contents

1. [How it works](#1-how-it-works)
2. [Requirements](#2-requirements)
3. [Quick start: two terminals](#3-quick-start-two-terminals)
4. [Running it step by step](#4-running-it-step-by-step)
5. [Talking to the robot](#5-talking-to-the-robot)
6. [Gestures and walking](#6-gestures-and-walking)
7. [Using it without a microphone](#7-using-it-without-a-microphone)
8. [Checking that everything works](#8-checking-that-everything-works)
9. [Stopping everything](#9-stopping-everything)
10. [Configuration](#10-configuration)
11. [Training the walking policy](#11-training-the-walking-policy)
12. [Troubleshooting](#12-troubleshooting)
13. [Development](#13-development)
14. [Project layout](#14-project-layout)
15. [Limitations and privacy](#15-limitations-and-privacy)

## 1. How it works

There are two parts, one per terminal. They talk to each other over ROS 2 topics.

- **Terminal 1, the simulator.** Isaac Sim simulates the R1 on a flat floor. A neural network policy, trained with reinforcement learning, moves the legs so the robot keeps its balance and walks at the speed it is told. The arms, waist and head follow the gesture commands.
- **Terminal 2, the assistant.** Five ROS 2 nodes listen, think, speak, gesture and walk.

| Node           | What it does                                                                                                                    |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `stt_node`     | Records the microphone, detects speech, transcribes it, and passes on what is said in a conversation started with the wake word. |
| `brain_node`   | Finds relevant knowledge and memories, asks Groq (or the local fallback) and gets back a reply, a gesture and a walking motion. |
| `tts_node`     | Speaks the reply. While it speaks, the microphone is muted so the robot does not hear itself.                                   |
| `gesture_node` | Turns the gesture name into smooth arm, waist and head movements, sent to the robot 50 times per second.                        |
| `motion_node`  | Turns a walking motion ("walk forward", "turn left") into walking speeds and stops when the robot has arrived.                  |

```text
 microphone --> stt_node --/voice/transcript--> brain_node --/voice/reply--> tts_node --> speakers
                   ^                              |      |                      |
                   +--------- /voice/speaking <---+------+----------------------+
                                                  |      |
                                     /robot/gesture      /robot/motion
                                                  v      v
                                      gesture_node      motion_node <-- /robot/odom --+
                                                  |      |                            |
                             /robot/joint_command |      | /cmd_vel                   |
                                                  v      v                            |
                                   Isaac Sim: Unitree R1 with the walking policy -----+
```

### ROS 2 topics

| Topic                   | Type                         | Published by   | Meaning                                            |
| ----------------------- | ---------------------------- | -------------- | -------------------------------------------------- |
| `/voice/transcript`     | `std_msgs/String`            | `stt_node`     | What you said (wake word removed)                  |
| `/voice/speech_ended`   | `std_msgs/Float64`           | `stt_node`     | When you stopped speaking (Unix time), for the timing lines in the logs |
| `/voice/reply`          | `std_msgs/String`            | `brain_node`   | What the robot will say                            |
| `/voice/speaking`       | `std_msgs/Bool`              | `tts_node`     | `true` while the robot is talking                  |
| `/voice/listening`      | `std_msgs/Bool`              | `stt_node`     | `false` while a sentence is being transcribed      |
| `/voice/mic_level`      | `std_msgs/Float32MultiArray` | `stt_node`     | `[current level, speech threshold]`, for tuning    |
| `/voice/reset`          | `std_msgs/String`            | you            | Clears the conversation history                    |
| `/voice/forget`         | `std_msgs/String`            | you            | Deletes the long-term memory                       |
| `/robot/gesture`        | `std_msgs/String`            | `brain_node`   | Gesture to play, for example `wave_right`          |
| `/robot/active_gesture` | `std_msgs/String`            | `gesture_node` | Gesture currently playing (`none` when idle)       |
| `/robot/joint_command`  | `sensor_msgs/JointState`     | `gesture_node` | Arm, waist and head joint targets                  |
| `/robot/motion`         | `std_msgs/String`            | `brain_node`   | Walking motion to perform, for example `turn_left` |
| `/robot/active_motion`  | `std_msgs/String`            | `motion_node`  | Motion currently running (`none` when idle)        |
| `/cmd_vel`              | `geometry_msgs/Twist`        | `motion_node`  | Walking speed and turning rate                     |
| `/robot/odom`           | `nav_msgs/Odometry`          | simulator      | Robot position, heading and speed                  |
| `/robot/joint_states`   | `sensor_msgs/JointState`     | simulator      | All 26 joint positions                             |
| `/clock`                | `rosgraph_msgs/Clock`        | simulator      | Simulation time                                    |

## 2. Requirements

This workspace was set up and tested with:

| Component          | Version or location                                                                                                     |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Operating system   | Ubuntu 24.04 LTS                                                                                                        |
| GPU                | NVIDIA RTX GPU (tested: RTX 4090 Laptop, 16 GB, driver 580)                                                             |
| Power              | Laptop on its charger and in **Performance** power mode (the voice is too slow otherwise)                               |
| ROS 2              | Jazzy, installed in `/opt/ros/jazzy`                                                                                    |
| Isaac Sim          | 6.2.0, pip install in `~/isaacsim-env`                                                                                  |
| Isaac Lab          | 3.0.0 in `~/IsaacLab`, installed into `~/isaacsim-env` (runs the walking policy)                                        |
| Robot model        | `~/g1_voice_ws/robots/r1/`: Unitree's R1 URDF and meshes, converted to USD in `usd/`                                    |
| Walking policy     | `~/g1_voice_ws/training/logs/rsl_rl/r1_walk_flat/` (see [Training the walking policy](#11-training-the-walking-policy)) |
| Ollama             | `~/.local/ollama`, with the fallback model `llama3.1:8b`                                                                |
| Groq               | An API key in the `GROQ_API_KEY` environment variable (optional)                                                        |
| Python environment | `~/g1_voice_ws/.venv` (faster-whisper, qwen-tts, nvidia-riva-client, sentence-transformers, PyTorch, colcon)                                |
| Models             | `~/g1_voice_ws/models/hf` (Whisper large-v3-turbo, Qwen3-TTS 0.6B, Qwen3-Embedding 0.6B)                                |
| Audio tools        | `arecord` and `aplay` (package `alsa-utils`, installed by default on Ubuntu)                                            |
| Internet           | For Groq, and when Isaac Sim starts                                                                                     |

To check the basics:

```bash
ls /opt/ros/jazzy/setup.bash ~/isaacsim-env/bin/isaacsim ~/.local/ollama/bin/ollama ~/g1_voice_ws/.venv/bin/python ~/g1_voice_ws/robots/r1/usd/r1.usd/R1/R1.usda
```

```bash
powerprofilesctl get
```

The last command should print `performance`. If it prints `power-saver` or `balanced`, switch to Performance in Ubuntu's Settings, Power.

### Groq API key

Set it once (it is read from the environment and never stored in the project):

```bash
echo 'export GROQ_API_KEY="your-key-here"' >> ~/.bashrc && source ~/.bashrc
```

Open new terminals after this, so they see the key. Without a key, the robot uses the local model only.

### NVIDIA Riva voices

The robot speaks with NVIDIA Riva's Magpie TTS voices, served by NVIDIA on [build.nvidia.com](https://build.nvidia.com/nvidia/magpie-tts-multilingual). Create a free API key there (sign in, open the model, "Get API Key") and set it once, like the Groq key:

```bash
echo 'export NVIDIA_API_KEY="nvapi-your-key-here"' >> ~/.bashrc && source ~/.bashrc
```

Without the key, or when NVIDIA cannot be reached, the robot speaks with Qwen3-TTS on this computer instead (after a failure it tries NVIDIA again 30 seconds later). The log says which voice is used:

```text
[tts_node]: TTS ready (NVIDIA Riva, voice Magpie-Multilingual.EN-US.Ray; local Qwen3-TTS if unreachable).
[tts_node]: NVIDIA_API_KEY is not set, so the Riva voice is off; using Qwen3-TTS.
```

List the voices and save a sample to listen to:

```bash
source ~/g1_voice_ws/.venv/bin/activate
python ~/g1_voice_ws/src/robot_voice_assistant/scripts/riva_voices.py --list
python ~/g1_voice_ws/src/robot_voice_assistant/scripts/riva_voices.py --voice Magpie-Multilingual.EN-US.Ray --out ray.wav && aplay ray.wav
```

Pick one with `riva_voice` in `config/voice_assistant.yaml`. A local Riva server (nothing sent to NVIDIA) needs the NVIDIA container toolkit for Docker, which needs administrator rights to install.

### Choosing the Groq model

Groq retired `llama-3.3-70b-versatile`. Run this to see the models your key can use:

```bash
curl -s https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY" | python3 -c "import json,sys; print('\n'.join(sorted(m['id'] for m in json.load(sys.stdin)['data'])))"
```

Measured with this assistant's 22 test questions (the chat tool, see [Check the conversation quality](#check-the-conversation-quality)):

| Model                 | Answer time | Notes                                                                                     |
| --------------------- | ----------- | ----------------------------------------------------------------------------------------- |
| `qwen/qwen3.8-27b`    | 1.0–1.8 s   | **Default.** Fastest, fuller and more natural answers, no hidden reasoning tokens          |
| `openai/gpt-oss-120b` | 1–7 s       | Largest model. A reasoning model: uses about 100 extra tokens per answer (effort set to low) |
| `openai/gpt-oss-20b`  | —           | Smaller reasoning model                                                                   |

Change it with `groq_model` in `config/voice_assistant.yaml`, or try one first with `chat --groq-model openai/gpt-oss-120b`.

On the free Groq plan each model allows **8,000 tokens per minute** and 1,000 requests per day. One question uses about 2,000 tokens (instructions, examples, knowledge and history), so a quick back-and-forth can hit the limit; the local model then answers for 30 seconds. A paid Groq plan removes this.

## 3. Quick start: two terminals

### Terminal 1: the robot in the simulator

```bash
~/g1_voice_ws/start_isaac.sh
```

A window called **Newton Viewer** opens with the R1 standing on the floor. Wait for these lines:

```text
[walk_scene] Walking. Listening on /cmd_vel and /robot/joint_command.
[walk_scene] simulation speed 55% of real time
```

### Terminal 2: the assistant (conversation, gestures and walking)

```bash
~/g1_voice_ws/start_assistant.sh
```

Wait for these lines (loading takes up to a minute the first time):

```text
[brain_node]: Language model: Groq qwen/qwen3.8-27b, falling back to local llama3.1:8b
[brain_node]: Brain ready (walking enabled).
[stt_node]: Listening. Say a wake word ('robot', 'r1', 'r one', 'are one') to talk to the robot.
[tts_node]: TTS ready (Qwen3-TTS, voice ryan, fast decoding).
```

Now talk to it, mixing conversation and movement freely:

- "Robot, what is the capital of India?"
- "Robot, come over here." (it walks about a metre towards you)
- "Robot, turn around."
- "Robot, wave at me."
- "Robot, tell me about Aideology."
- "Stop!" (no wake word needed: the conversation is still on)
- "Goodbye." (ends the conversation; say "robot" to start a new one)

To stop, press `Ctrl+C` in each terminal.

## 4. Running it step by step

The two start scripts run the commands below.

### Terminal 1 in detail

```bash
source /opt/ros/jazzy/setup.bash
```

```bash
source ~/isaacsim-env/bin/activate
```

```bash
cd ~/g1_voice_ws/training && python -u walk_scene.py
```

Options:

| Option               | Meaning                                                                |
| -------------------- | ---------------------------------------------------------------------- |
| `--viewer newton_gl` | Light OpenGL viewer (default, about 55% of real time)                  |
| `--viewer kit`       | Full Isaac Sim window (about 4% of real time on this laptop: too slow) |
| `--headless`         | No window (about 85% of real time)                                     |
| `--checkpoint PATH`  | Use a specific trained policy (`model_<iteration>.pt`)                 |

For example: `~/g1_voice_ws/start_isaac.sh --headless`. Movements and gestures follow simulation time, so they stay correct when the simulation runs slower than real time, just in slow motion. In the viewer, drag with the mouse to rotate the camera and scroll to zoom.

### Terminal 2 in detail

`start_assistant.sh` first starts Ollama (the local fallback model) if it is not running. To do it yourself, run this in its own terminal and leave it running:

```bash
~/.local/ollama/bin/ollama serve
```

Then, in terminal 2, in this order: ROS 2, then the Python environment (Whisper, Qwen3-TTS, Qwen3-Embedding), then this workspace, then the launch command:

```bash
source /opt/ros/jazzy/setup.bash
```

```bash
source ~/g1_voice_ws/.venv/bin/activate
```

```bash
source ~/g1_voice_ws/install/setup.bash
```

```bash
ros2 launch robot_voice_assistant voice_assistant.launch.py
```

Launch arguments (they also work with `start_assistant.sh`):

| Argument                     | Default        | Meaning                                                               |
| ---------------------------- | -------------- | --------------------------------------------------------------------- |
| `use_microphone:=false`      | `true`         | Do not start the microphone; send text on `/voice/transcript` instead |
| `use_sim_time:=false`        | `true`         | Time gestures and motions with the wall clock instead of `/clock`     |
| `config:=/path/to/file.yaml` | package config | Use a different settings file                                         |

## 5. Talking to the robot

1. Make sure both terminals are running (see [Quick start](#3-quick-start-two-terminals)).
2. Speak clearly, about an arm's length from the laptop microphone.
3. Start with the wake word: "Robot, what is the capital of India?". You can also say just "Robot", pause, and then ask; if no question follows within 3 seconds, it says hello. "Robot" is recognised most reliably; "R1" is often heard as "are we" or "oh one".
4. Wait for the answer. While the robot speaks, it does not listen.
5. Keep talking without the wake word: the conversation lasts until you end it with "goodbye", "stop listening", "that's all" or "go to sleep". "Stop" on its own stops walking and keeps the conversation going.

The log shows when a conversation starts and ends:

```text
[stt_node]: Conversation started: no wake word needed until you say goodbye.
[stt_node]: Conversation ended. Say a wake word ('robot', 'r1', 'r one', 'are one') to start again.
```

The assistant terminal shows what it understood and decided:

```text
[stt_node]: Heard: come over here
[brain_node]: [groq] Reply: "Sure, I'm coming over to you now." gesture=wave_right motion=walk_forward
[motion_node]: Motion: walk_forward
[gesture_node]: Playing 'wave_right' (3.8s)
```

### What you can ask

The robot uses a general-purpose language model, so you can ask about almost anything: general knowledge, science, history, maths, advice, small talk and jokes. In the same conversation you can ask it to move.

- **Facts**: "What is the capital of India?" gets "The capital of India is New Delhi."
- **Follow-ups**: "And what about France?" (it remembers the last 10 questions and answers).
- **Calculations**: "What is 23 times 17?" (done by a built-in calculator, always right).
- **Aideology and your own documents**: answered from the knowledge folder (see [Knowledge base, memory and command matching](#knowledge-base-memory-and-command-matching)).
- **Today's date**: it is given the computer's date and time.
- **Things that change**: it says what it last knew and that it may have changed.
- **Live information** (weather, news, prices): it says it cannot check.
- **About itself**: it knows it is a Unitree R1 with arms, a waist and a head but no hands, and that it can walk forward and backward and turn, but not step sideways, jump or pick things up.

Answers are kept to at most four sentences, because they are spoken. Ask "tell me more" for details.

## 6. Gestures and walking

The language model picks a gesture with every reply, and a walking motion when you ask the robot to move. Both can happen at the same time, for example waving while walking towards you.

### Gestures

The R1 has no hands, so gestures use the arms (5 joints each), the waist (roll and yaw) and the neck (head pitch and yaw):

| Gesture                   | Movement                                                    |
| ------------------------- | ----------------------------------------------------------- |
| `wave_right`, `wave_left` | Raises an arm beside the head and waves                     |
| `nod`                     | Nods the head twice ("yes")                                 |
| `shake`                   | Shakes the head ("no")                                      |
| `bow`                     | Bows the head with the arms together in front               |
| `point_forward`           | Points straight ahead with the right arm                    |
| `celebrate`               | Raises both arms in a V and pumps them, looking up          |
| `shrug`                   | Opens the forearms and tilts the head back ("I don't know") |
| `think`                   | Raises the right arm to the chin and tilts the head         |
| `open_arms`               | Opens both arms wide                                        |
| `look_left`, `look_right` | Turns the head and upper body to one side                   |
| `none`                    | Stands relaxed                                              |

The simulation does not stop an arm from passing through the robot's own head or body, so every gesture is checked in advance with the R1's real geometry (from Unitree's URDF and meshes): the tests play each gesture, and every switch from one gesture to another halfway through, and require at least 4 cm between the arms and the head and 3 cm to the chest.

### Walking motions

| Motion          | What the robot does               |
| --------------- | --------------------------------- |
| `walk_forward`  | Walks forward about 1 metre       |
| `walk_backward` | Walks backward about half a metre |
| `turn_left`     | Turns 90 degrees to the left      |
| `turn_right`    | Turns 90 degrees to the right     |
| `turn_around`   | Turns 180 degrees                 |
| `keep_walking`  | Walks forward until told to stop  |
| `stop`          | Stops                             |

`motion_node` executes a motion using the robot's odometry, so "walk forward" really stops after about one metre and "turn left" after 90 degrees. A motion that cannot finish (for example because the robot is stuck) gives up after a time limit. If no walking command arrives for half a second, the robot stops.

### How well it walks

Measured with the final policy (iteration 1500) in terminal 1's scene, headless at 100% of real time. The robot did not fall in any test: the pelvis always stayed between 0.70 and 0.73 m (0.72 m when standing).

Named motions, as used by the voice assistant:

| Motion                      | Target      | Measured                                 |
| --------------------------- | ----------- | ---------------------------------------- |
| `walk_forward`              | 1.0 m       | 1.07 to 1.08 m, heading within 3 degrees |
| `walk_backward`             | 0.5 m       | 0.40 m                                   |
| `turn_left`                 | 90 degrees  | 83 degrees                               |
| `turn_right`                | 90 degrees  | 84 degrees                               |
| `turn_around`               | 180 degrees | 190 degrees                              |
| `walk_forward` while waving | 1.0 m       | 1.07 m, steady                           |
| `keep_walking`, then `stop` | until told  | walked 2.4 m, then stopped               |

Speed commands on `/cmd_vel` (measured over 4 seconds once walking):

| Command                            | Measured                                           |
| ---------------------------------- | -------------------------------------------------- |
| Forward 1.0 / 0.6 / 0.3 m/s        | 0.85 / 0.52 / 0.24 m/s                             |
| Backward 0.3 m/s                   | 0.20 m/s                                           |
| Turn left 0.8 / 0.4 rad/s          | 0.97 / 0.52 rad/s                                  |
| Turn right 0.8 / 0.4 rad/s         | 0.88 / 0.29 rad/s                                  |
| Forward 0.5 m/s and turn 0.5 rad/s | 0.45 m/s and 0.46 rad/s                            |
| Sideways 0.25 m/s (left / right)   | 0.07 / 0.00 m/s: the policy does not walk sideways |

The robot keeps moving for a moment after the command stops (about 0.3 m after walking at 0.5 m/s, 13 to 22 degrees after a turn), so `motion_node` stops early by the distance covered in 0.5 s (walking) or the angle turned in 0.33 s (turning). Side steps are not offered, because the policy does not walk sideways; asked to step sideways, the robot says it cannot yet and offers to turn instead.

### Commanding the robot directly

Without the voice assistant (with terminal 1 running and `motion_node` running in terminal 2):

```bash
ros2 topic pub --once /robot/motion std_msgs/msg/String "{data: turn_left}"
```

```bash
ros2 topic pub --once /robot/gesture std_msgs/msg/String "{data: celebrate}"
```

Or drive it with a speed (forward m/s and turning rad/s), repeated 20 times per second:

```bash
ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5}, angular: {z: 0.0}}"
```

## 7. Using it without a microphone

### Chat with the robot's brain by typing

No simulator, microphone or speakers needed. This uses exactly the same brain as the robot:

```bash
source /opt/ros/jazzy/setup.bash && source ~/g1_voice_ws/install/setup.bash
```

```bash
ros2 run robot_voice_assistant chat
```

```text
You: Can you come over here?
Robot [llama, wave_right, walk: walk_forward, 0.5s]: I'm coming over to you now.

You: What is the capital of Japan?
Robot [llama, nod, 0.7s]: The capital of Japan is Tokyo. It is one of the largest cities in the world.
```

Each answer shows which model answered (`groq`, `llama`, or `commands` when no model was needed), the gesture, the walking motion and the time taken. Type `/reset` to start a new conversation, `/forget` to delete the long-term memory and `/quit` (or `Ctrl+D`) to exit.

| Option                  | Meaning                                                                  |
| ----------------------- | ------------------------------------------------------------------------ |
| `--local`               | Use only the local fallback model, not Groq                              |
| `--fallback-model NAME` | Use another local Ollama model, for example `--fallback-model mistral`   |
| `--knowledge-dir PATH`  | Use another knowledge folder                                             |
| `--no-memory`           | Do not read or write the long-term memory                                |
| `--no-embeddings`       | No knowledge search, memory or command matching                          |
| `--questions FILE`      | Answer every line of a file; a line with `---` starts a new conversation |

### Send sentences to the running assistant

Start terminal 2 without the microphone:

```bash
~/g1_voice_ws/start_assistant.sh use_microphone:=false
```

In a third terminal (after `source /opt/ros/jazzy/setup.bash`), send a sentence as if you had said it. No wake word is needed here:

```bash
ros2 topic pub --once /voice/transcript std_msgs/msg/String "{data: 'Can you come over here?'}"
```

Make the robot say something:

```bash
ros2 topic pub --once /voice/reply std_msgs/msg/String "{data: 'Hello, I am R1.'}"
```

## 8. Checking that everything works

Run these in a third terminal, after `source /opt/ros/jazzy/setup.bash`.

All topics should be there, including `/clock`, `/cmd_vel`, `/robot/odom`, `/robot/joint_command` and the `/voice/...` topics:

```bash
ros2 topic list
```

The simulator is running:

```bash
ros2 topic hz /clock
```

All five assistant nodes (`/brain_node`, `/gesture_node`, `/motion_node`, `/stt_node`, `/tts_node`) and the simulator's `/robot_walk_scene` are running:

```bash
ros2 node list
```

The robot's position (it changes when it walks):

```bash
ros2 topic echo /robot/odom --field pose.pose.position
```

The microphone hears you: when you speak, the first number must be clearly larger than the second:

```bash
ros2 topic echo /voice/mic_level --field data
```

The local fallback model is loaded:

```bash
~/.local/ollama/bin/ollama ps
```

## 9. Stopping everything

1. Press `Ctrl+C` in terminal 2. This stops the five nodes and releases the microphone.
2. Press `Ctrl+C` in terminal 1, or close the viewer window.
3. If `start_assistant.sh` started Ollama, it keeps running in the background and uses GPU memory while a model is loaded. Stop it with:

   ```bash
   pkill -f "ollama serve"
   ```

To make sure nothing is left running (no output means everything is stopped):

```bash
pgrep -af "robot_voice_assistant|walk_scene|arecord|ollama"
```

## 10. Configuration

All assistant settings are in `src/robot_voice_assistant/config/voice_assistant.yaml`. Changes take effect the next time you start the assistant.

### Speech recognition (`stt_node`)

| Setting           | Default                               | Meaning                                                                                    |
| ----------------- | ------------------------------------- | ------------------------------------------------------------------------------------------ |
| `model_size`      | `large-v3-turbo`                      | Whisper model: `large-v3-turbo` (most accurate), `small.en` (faster, less GPU memory)      |
| `device`          | `cuda`                                | `cuda` for the GPU, `cpu` without a GPU                                                    |
| `mic_device`      | `default`                             | Microphone for `arecord`; list devices with `arecord -l`, for example `plughw:1,6`         |
| `min_rms`         | `0.01`                                | Lowest sound level that can count as speech                                                |
| `threshold_ratio` | `2.5`                                 | How much louder than background noise speech must be; lower it if the robot does not react |
| `wake_words`      | `["robot", "r1", "r one", "are one"]` | Words that start a conversation; `[""]` answers everything (not recommended in a shared room) |
| `wake_word_sound_alikes` | `["robert"]`                     | What Whisper sometimes writes for "robot"; counts only at the start of a sentence          |
| `conversation_idle_timeout_s` | `0.0`                     | `0` = a conversation lasts until you say goodbye; e.g. `300` ends it after 5 quiet minutes |
| `wake_word_wait_s` | `3.0`                                | After the wake word alone, how long to wait for the question before saying hello           |
| `end_silence_ms`  | `700`                                 | Silence that ends a sentence; `900` hears slightly better in loud noise but answers 0.2 s later |
| `vocabulary`      | `["Aideology", "Jorge Colotto", ...]` | Names Whisper should expect, given to it as a prompt; add your own names here              |
| `beam_size`       | `5`                                   | Whisper beam search width                                                                  |
| `recorder`        | `arecord`                             | Program that records the microphone (`scripts/fake_microphone.py` plays test speech)       |
| `unmute_delay_s`  | `0.5`                                 | Pause after the robot stops speaking before it listens again                               |

### Language model (`brain_node`)

| Setting          | Default                   | Meaning                                                                  |
| ---------------- | ------------------------- | ------------------------------------------------------------------------ |
| `use_groq`       | `true`                    | Use Groq in the cloud (needs `GROQ_API_KEY`); `false` = local model only |
| `groq_model`     | `qwen/qwen3.8-27b`        | Groq model (see [Choosing the Groq model](#choosing-the-groq-model))     |
| `fallback_model` | `llama3.1:8b`             | Local Ollama model that answers when Groq is unavailable                 |
| `ollama_url`     | `http://localhost:11434`  | Address of the Ollama server                                             |
| `robot_name`     | `R1`                      | Name the robot uses for itself                                           |
| `history_turns`  | `10`                      | How many earlier question and answer pairs it remembers word for word    |

Every question goes to Groq first. If Groq cannot answer (no internet, no or an invalid `GROQ_API_KEY`, or Groq's rate limit), the local `llama3.1:8b` answers instead, and Groq is tried again after 30 seconds (or after the time Groq asks for when it rate-limits). The assistant terminal shows which model answered (`[groq]` or `[llama]`).

### Knowledge base, memory and command matching

Also in `brain_node`. Qwen3-Embedding turns text into vectors that capture its meaning and is used three ways:

| Setting                   | Default                                             | Meaning                                                                     |
| ------------------------- | --------------------------------------------------- | --------------------------------------------------------------------------- |
| `use_embeddings`          | `true`                                              | `false` turns off all three features below                                  |
| `embedding_model`         | `Qwen/Qwen3-Embedding-0.6B`                         | Embedding model                                                             |
| `knowledge_dir`           | `""`                                                | Folder of `.md` and `.txt` files; empty means `config/knowledge/`           |
| `knowledge_min_score`     | `0.40`                                              | How closely a passage must match the question to be used                    |
| `use_memory`              | `true`                                              | Remember earlier conversations                                              |
| `memory_file`             | `~/.local/share/robot_voice_assistant/memory.jsonl` | Where the memories are stored                                               |
| `memory_min_score`        | `0.40`                                              | How closely a memory must match the question to be recalled                 |
| `command_match_threshold` | `0.68`                                              | How clearly a sentence must be a command to be recognised without the model |

- **Knowledge base.** Every `.md` and `.txt` file in `src/robot_voice_assistant/config/knowledge/` is split into passages (one per line or bullet point, or a few sentences of running text). For each question the best-matching passages are given to the language model. Headings and lines ending in `:` (such as `About Aideology:`) are used as context for the lines below them. Add files, rebuild (`colcon build --symlink-install`) and restart; passages are embedded once and cached until a file changes. The current Aideology facts were collected from public company listings; please check that they are correct.
- **Long-term memory.** Every exchange (except small talk) is saved to `memory_file`. When you later ask something related ("What is my name?"), the most relevant earlier exchanges are given to the language model. Delete all memories with `ros2 topic pub --once /voice/forget std_msgs/msg/String "{}"`, or `/forget` in the typed chat.
- **Command matching.** Clear commands such as "turn left" or "wave at me" are recognised by meaning. This adds a motion or gesture when the language model forgot one, and lets the robot follow clear commands even when no language model answers. It never overrides the language model's choice, and its threshold is strict: with lower values, ordinary sentences such as "My left hand hurts" were mistaken for commands.

### How the answers are shaped

In `src/robot_voice_assistant/robot_voice_assistant/conversation.py`:

- `build_system_prompt`: the rules (accurate, short, spoken style, no lists, admit uncertainty, always move when asked to, what the robot can and cannot do).
- `WALKING_EXAMPLES`: example exchanges the model copies for tone, length, gestures and motions. Adding an example is the most effective way to change how it answers a kind of question.
- Code checks after the model answers: replies cut off mid-word are retried, replies longer than four sentences are shortened, calculations are done by `arithmetic.py`, and missed commands are filled in by command matching.

### Voice (`tts_node`)

| Setting       | Default                                         | Meaning                                                               |
| ------------- | ----------------------------------------------- | --------------------------------------------------------------------- |
| `tts_engine`  | `riva`                                          | `riva` (NVIDIA Riva voice in the cloud, Qwen3-TTS as fallback), `qwen3` (Qwen3-TTS on this computer) or `piper` (instant, more robotic) |
| `riva_voice`  | `Magpie-Multilingual.EN-US.Ray`                 | Riva voice and style. English voices: Jason, Leo, Ray, Diego, Pascal (male); Mia, Aria, Sofia, Isabela, Louise (female). Jason, Mia, Aria, Leo, Sofia and Ray also have styles such as `.Calm`, `.Neutral`, `.Sad` (only Jason has `.Happy`); list them with `scripts/riva_voices.py --list` |
| `riva_language` | `en-US`                                       | Language of the Riva voice                                            |
| `riva_sample_rate` | `44100`                                    | Sample rate requested from Riva                                       |
| `riva_speaking_rate` | `0.9`                                    | Speed for the Riva voice (`speaking_rate` is for Qwen3-TTS); Ray is about 224 words per minute at 1.0 and 200 at 0.9 |
| `qwen_model`  | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`          | Qwen3-TTS model                                                       |
| `speaker`     | `ryan`                                          | English voices: `aiden`, `dylan`, `eric`, `ryan`                      |
| `language`    | `english`                                       | Language of the replies                                               |
| `instruct`    | `""`                                            | Speaking style, for example `"Speak in a warm, friendly tone."`       |
| `fast_decoding` | `true`                                        | Faster Qwen3-TTS decoding (CUDA graph); `false` = the library's own   |
| `voice_model` | `~/g1_voice_ws/models/en_US-lessac-medium.onnx` | Piper voice file (only with `tts_engine: piper`)                      |
| `player`      | `aplay`                                         | Program that plays the audio                                          |
| `speaking_rate` | `1.1`                                         | How fast it speaks: `1.0` = the voice's own pace, `1.1` = normal conversation (about 165 words per minute), `1.2` = brisk; the pitch stays the same |

Before playing, the silence Qwen3-TTS puts around every piece (about half a second each side) is trimmed to a short pause, and speech is sped up by `speaking_rate` without changing its pitch (`audio_effects.py`). Together that takes a reply from about 123 to about 167 words per minute; Whisper still understood every word at 1.3x. Style instructions such as "speak faster" make no difference to the 0.6B voice model.

Replies are spoken in pieces: the first piece is kept short (one short sentence, or a long sentence cut at a comma or a word like "that"), and each next piece is generated while the previous one plays. The robot starts talking about 1.5 seconds after it has its answer.

`fast_decoding` replaces the way Qwen3-TTS predicts the 15 detail codes of every 1/12 second of audio: the library runs a full Hugging Face `generate()` call for each, which is mostly Python overhead. The replacement (`fast_code_predictor.py`) runs the same model in a fixed-size loop captured once as a CUDA graph. It picks the same codes as the library (checked by the tests), and halves the time.

Time to generate one second of speech on the RTX 4090 Laptop GPU (Performance power mode, charger connected):

| Decoding                        | Time per second of speech |
| ------------------------------- | ------------------------- |
| `fast_decoding: true` (default) | 0.35 s                    |
| `fast_decoding: false`          | 0.7 s                     |

In Power saver mode or on battery it is 4 to 6 times slower, which causes long pauses.

### Where the time goes

Every log line starts with the local time to the millisecond, and each step logs how long it took. One question in terminal 2 looks like this:

```text
[INFO] [2026-10-27 09:50:40.535] [stt_node]: Heard: Robot, what is the capital of India? (3.1 s of speech; waited 0.69 s for silence, speech to text 0.31 s)
[INFO] [2026-10-27 09:50:42.646] [brain_node]: [groq] Reply: 'The capital of India is New Delhi. ...' gesture=nod motion=none
[INFO] [2026-10-27 09:50:42.646] [brain_node]: Timing: knowledge search 0.02 s, language model 1.06 s, command matching 0.01 s, memory write 0.01 s, total 1.11 s (groq); reply ready 2.11 s after you stopped speaking
[INFO] [2026-10-27 09:50:43.798] [tts_node]: Voice started 1.15 s after the reply arrived; 3.26 s after you stopped speaking
[INFO] [2026-10-27 09:50:49.128] [tts_node]: Finished speaking after 7.48 s; voice generated in 1.15 s (34 characters), 1.33 s (42 characters)
```

- `stt_node`: the wait for 0.69 s of silence that ends your sentence, and the Whisper time.
- `brain_node`: each step of answering, which model answered (`groq`, `llama` or `commands`), and the time since you stopped speaking.
- `tts_node`: when the voice started, and how long each spoken piece took to generate. If a reply had to wait for the previous one to finish, the wait is shown too.

"After you stopped speaking" uses the `/voice/speech_ended` topic from `stt_node`, so it is left out for typed requests. To use your own log format, set `RCUTILS_CONSOLE_OUTPUT_FORMAT` before starting the assistant.

Typical times, measured from the moment you stop speaking until the robot's voice starts:

| Step                                                    | Time       |
| ------------------------------------------------------- | ---------- |
| Waiting for the end of your sentence (0.7 s of silence) | 0.7 s      |
| Speech to text (Whisper large-v3-turbo)                 | 0.2–0.6 s  |
| Knowledge and memory search (Qwen3-Embedding)           | 0.03 s     |
| Answer from Groq (`qwen/qwen3.8-27b`)                   | 1.1–1.6 s (sometimes up to 8 s) |
| Answer from the local `llama3.1:8b` (when Groq is rate limited) | 4–8 s |
| First spoken piece (Qwen3-TTS)                          | 1.3–1.8 s  |
| **Total**                                               | **about 3.5–4.5 s** |

### Gestures and motions (`gesture_node`, `motion_node`)

| Setting         | Default                | Meaning                                               |
| --------------- | ---------------------- | ----------------------------------------------------- |
| `rate_hz`       | `50.0` / `20.0`        | How often joint targets (gestures) or speeds are sent |
| `command_topic` | `/robot/joint_command` | Topic the simulator listens on (`gesture_node` only)  |

The gestures are defined in `robot_voice_assistant/gestures.py` and the motions in `robot_voice_assistant/motions.py`.

## 11. Training the walking policy

The walking policy was trained with Isaac Lab 3.0 on the task `R1-Walk-Flat`, defined in `~/g1_voice_ws/training/walk_task/`:

- the Unitree R1 from `robot_cfg.py` (converted from Unitree's URDF; motor limits from the URDF);
- PhysX physics, like the scene in terminal 1;
- the policy controls the 12 leg joints; the waist, arms and head are moved through the gesture poses during training, so the robot learns to keep its balance while it gestures;
- forward, backward, sideways and turning speed commands, and standing still (the policy did not learn to walk sideways; see [How well it walks](#how-well-it-walks)).

To train again (about an hour on the RTX 4090 Laptop GPU with 4096 robots in parallel and the GPU busy half of the time):

```bash
nice -n 15 ~/g1_voice_ws/training/train_walk.sh --num_envs 4096 --gpu_duty 0.5
```

`--gpu_duty 0.5` lets the GPU work only half of the time, so the computer stays usable (leave it out to train about twice as fast). Checkpoints are saved every 50 iterations in `~/g1_voice_ws/training/logs/rsl_rl/r1_walk_flat/<date>/model_<iteration>.pt`; terminal 1 loads the newest checkpoint of the newest run. Pressing `Ctrl+C` in the training terminal stops training and keeps the checkpoints. To continue a stopped run (here from iteration 750 to 1500):

```bash
~/g1_voice_ws/training/train_walk.sh --num_envs 4096 --max_iterations 750 --resume_optimizer_state --checkpoint ~/g1_voice_ws/training/logs/rsl_rl/r1_walk_flat/<date>/model_750.pt
```

Watch the training curves (then open `http://localhost:6006`); the most useful are the mean episode length (how long the robots stay up, at most 1000 steps) and the velocity tracking rewards:

```bash
~/isaacsim-env/bin/tensorboard --logdir ~/g1_voice_ws/training/logs/rsl_rl/r1_walk_flat
```

Watch 32 robots walk with random commands:

```bash
~/g1_voice_ws/training/play_walk.sh
```

After changing the gestures, export the poses again before training:

```bash
~/g1_voice_ws/.venv/bin/python ~/g1_voice_ws/training/export_upper_body_poses.py
```

To rebuild the robot model from Unitree's URDF (only needed if the URDF changes):

```bash
source ~/isaacsim-env/bin/activate && cd ~/IsaacLab && python scripts/tools/convert_urdf.py ~/g1_voice_ws/robots/r1/R1.urdf ~/g1_voice_ws/robots/r1/usd/r1.usd --merge_joints --viz none
```

## 12. Troubleshooting

| Problem                                      | Fix                                                                                                                                                                            |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Long pauses before or while the robot speaks | The laptop is not in Performance power mode or is on battery. Switch to Performance, or set `tts_engine: piper`.                                                               |
| The robot does not move                      | Terminal 1 must show `Walking.`; check `ros2 topic hz /clock`. `motion_node` needs `/robot/odom` from terminal 1.                                                              |
| The robot does not react when I speak        | Say a wake word. Check `ros2 topic echo /voice/mic_level --field data` and lower `threshold_ratio` if your level stays below the threshold. Check the microphone is not muted. |
| It reacts to other people in the room        | Keep the wake words on, and raise `threshold_ratio`.                                                                                                                           |
| `GROQ_API_KEY is not set` in the log         | Set the key (see [Groq API key](#groq-api-key)) and start the assistant from a new terminal.                                                                                   |
| `Groq unavailable (HTTPError: HTTP 401)`     | The key is wrong or expired. Create a new one in the Groq console and update `~/.bashrc`.                                                                                      |
| `Groq unavailable (RateLimitedError ...)`    | Over the plan's 8,000 tokens per minute (about 4 questions a minute). The local model answers until Groq accepts requests again; see [Choosing the Groq model](#choosing-the-groq-model). |
| `Groq unavailable (HTTPError: HTTP 404)`     | The model was retired. Pick one from the list in [Choosing the Groq model](#choosing-the-groq-model) and set `groq_model`.                                                       |
| `Could not reach Ollama`                     | Start it with `~/.local/ollama/bin/ollama serve`.                                                                                                                              |
| `model "llama3.1:8b" not found`              | Download it: `~/.local/ollama/bin/ollama pull llama3.1:8b`.                                                                                                                    |
| An answer is wrong or outdated               | Add the correct fact to `config/knowledge/`.                                                                                                                                   |
| `Microphone stream ended`                    | `arecord` cannot open the device. Run `arecord -l` and set `mic_device`.                                                                                                       |
| The full Isaac Sim window is extremely slow  | Use the default light viewer (`--viewer newton_gl`); the full window runs at about 4% of real time on this laptop.                                                             |
| `No checkpoint found` in terminal 1          | Train the walking policy first (see [Training the walking policy](#11-training-the-walking-policy)).                                                                           |
| `ros2: command not found`                    | Run `source /opt/ros/jazzy/setup.bash` first.                                                                                                                                  |
| `Package 'robot_voice_assistant' not found`  | Build the workspace (see [Development](#13-development)) and `source ~/g1_voice_ws/install/setup.bash`.                                                                        |

Logs: Isaac Sim and Isaac Lab in `~/isaacsim-env/lib/python3.12/site-packages/isaacsim/kit/logs/Kit/`, the ROS 2 nodes in `~/.ros/log/`, Ollama (when started by the script) in `~/g1_voice_ws/log/ollama.log`.

## 13. Development

### Build the workspace

Needed after adding or renaming files (changes to existing Python files are picked up without rebuilding):

```bash
cd ~/g1_voice_ws && source /opt/ros/jazzy/setup.bash && source .venv/bin/activate && colcon build --symlink-install
```

### Run the tests

The tests cover speech detection, the wake word check, the language model backends and fallback, reading the model's answers, the calculator, the knowledge base, memory and command matching, the speech pipeline, the walking motions and the gestures (joint limits, smooth movement, and the arm clearance from the head and chest). Run them in a terminal where ROS 2 is **not** sourced (a ROS 2 pytest plugin conflicts), for example with `env -i`:

```bash
cd ~/g1_voice_ws/src/robot_voice_assistant && env -i HOME=$HOME PATH=/usr/bin:/bin ../../.venv/bin/python -m pytest test
```

### Check speech recognition

`scripts/check_speech_recognition.py` speaks 26 test sentences in 6 Qwen3-TTS voices at different loudness, adds a quiet room, a fan or loud noise, and streams them through the same steps as `stt_node` (speech detection, Whisper with the vocabulary prompt, the wake word and conversation check). It reports how many came out exactly right:

```bash
source /opt/ros/jazzy/setup.bash && source ~/g1_voice_ws/.venv/bin/activate
python ~/g1_voice_ws/src/robot_voice_assistant/scripts/check_speech_recognition.py --show-transcripts
```

Results on the RTX 4090 Laptop GPU (468 sentences):

| Setup                                      | Quiet room  | Fan         | Loud noise  | All       | Word error rate |
| ------------------------------------------ | ----------- | ----------- | ----------- | --------- | --------------- |
| Before (no vocabulary, 8-second follow-up) | —           | —           | —           | 45.1%     | 24.0%           |
| Without the vocabulary prompt              | 97/156      | 94/156      | 88/156      | 59.6%     | 14.4%           |
| **With the vocabulary prompt (default)**   | **146/156** | **144/156** | **131/156** | **90.0%** | **5.2%**        |

It also counts names and walking commands that appear in a transcript without being said, because Whisper can copy words from its prompt into noisy speech: none with the default settings. That is why the prompt contains only names and no commands.

What it still gets wrong: "R1" as a wake word (heard as "are we", "oh one"), "Aideology" sometimes as "ideology" (the brain is told to read it as Aideology), "turn" as "tune" (the brain still understands it), and a quiet voice in loud noise.

### Test whole conversations

`scripts/fake_microphone.py` stands in for the microphone, and `scripts/check_conversation_live.py` speaks whole conversations into the running assistant, waits for every reply like a person would, and checks what was heard, the reply, the gesture, the motion and whether the conversation is still on. Start the assistant with the fake microphone and silent speakers, in one terminal:

```bash
mkdir -p /tmp/robot_mic && sed -e "s|    recorder: arecord .*|    recorder: \"python3 $HOME/g1_voice_ws/src/robot_voice_assistant/scripts/fake_microphone.py --spool /tmp/robot_mic --noise 0.01\"|" -e "s|    player: aplay|    player: /bin/true|" ~/g1_voice_ws/src/robot_voice_assistant/config/voice_assistant.yaml > /tmp/robot_mic/test.yaml
```

```bash
~/g1_voice_ws/start_assistant.sh config:=/tmp/robot_mic/test.yaml
```

`--noise` sets the background noise (0.0015 quiet room, 0.01 fan, 0.02 loud). The conversations use the real long-term memory file unless you also change `memory_file` in the copy.

and run the conversations in a second terminal:

```bash
source /opt/ros/jazzy/setup.bash && source ~/g1_voice_ws/.venv/bin/activate && source ~/g1_voice_ws/install/setup.bash
python ~/g1_voice_ws/src/robot_voice_assistant/scripts/check_conversation_live.py --spool /tmp/robot_mic
```

### Check the conversation quality

Question sets are in `src/robot_voice_assistant/test/conversations/`. Run one and read the answers, for example after changing the model, the rules or the examples:

```bash
ros2 run robot_voice_assistant chat --no-memory --questions ~/g1_voice_ws/src/robot_voice_assistant/test/conversations/walking_questions.txt
```

### Rebuild the arm collision model

`config/robot_collision_model.npz` (used by the gesture tests) is built from the R1's URDF and meshes:

```bash
cd ~/g1_voice_ws/src/robot_voice_assistant && ../../.venv/bin/python scripts/build_collision_model.py ~/g1_voice_ws/robots/r1/R1.urdf ~/g1_voice_ws/robots/r1
```

## 14. Project layout

```text
~/g1_voice_ws/
├── README.md                      this file
├── start_isaac.sh                 terminal 1: the R1 in the simulator
├── start_assistant.sh             terminal 2: conversation, gestures and walking
├── .venv/                         Python environment for the assistant
├── models/                        Whisper, Qwen3-TTS, Qwen3-Embedding and Piper models
├── robots/r1/                     Unitree R1 URDF, meshes and the converted USD
├── training/                      the walking policy (Isaac Lab)
│   ├── walk_task/                 training task: robot config, rewards, gesture poses
│   ├── train_walk.sh              trains the policy
│   ├── play_walk.sh               shows 32 robots walking
│   ├── walk_scene.py              terminal 1's scene: runs the policy, connects to ROS 2
│   ├── walk_ros_bridge.py         ROS 2 topics of the scene
│   └── logs/                      training runs and checkpoints
└── src/robot_voice_assistant/     the ROS 2 package
    ├── config/voice_assistant.yaml          all settings
    ├── config/knowledge/                    documents the robot can search
    ├── config/robot_collision_model.npz     R1 arm, head and torso shapes for the gesture tests
    ├── launch/voice_assistant.launch.py     starts the five nodes
    ├── scripts/build_collision_model.py     rebuilds the collision model
    ├── robot_voice_assistant/
    │   ├── stt_node.py, brain_node.py, tts_node.py, gesture_node.py, motion_node.py   the ROS 2 nodes
    │   ├── chat_cli.py            type to the robot's brain
    │   ├── assistant_factory.py   builds the brain from the settings
    │   ├── conversation.py        rules, examples, memory and answer checks
    │   ├── llm_backends.py        Groq client and the local fallback
    │   ├── ollama_client.py       local Ollama server
    │   ├── semantic.py            knowledge base, long-term memory, command matching
    │   ├── speech_pipeline.py     Qwen3-TTS and Piper, sentence-by-sentence speaking
    │   ├── gestures.py            R1 gestures and joint limits
    │   ├── motions.py             walking motions on odometry
    │   ├── kinematics.py          R1 arm kinematics for the clearance checks
    │   ├── vad.py, wake_word.py, transcript_filter.py   listening
    │   ├── arithmetic.py          calculator
    │   └── cuda_libs.py, node_runner.py                 helpers
    └── test/                      unit tests and conversation question sets
```

## 15. Limitations and privacy

### Limitations

- The robot walks on flat ground only: forward, backward and turning. It does not step sideways, cannot see obstacles, and cannot run, jump or climb.
- The R1 has no hands and cannot pick things up. Gestures are hand-made movements of the arms, waist and head, carried out by the physics simulation through the joint motors.
- The walking policy was trained in simulation only.
- Neither language model can look things up; they know nothing after their training data ends.
- English only.

### Privacy

- **With a Groq key, what you say to the robot is sent to Groq** (everything said during a conversation, from the wake word until "goodbye"), together with the relevant knowledge passages and memories. Without a key, or with `use_groq: false`, everything runs on this computer.
- **With an NVIDIA key, the robot's replies (not what you say) are sent to NVIDIA** to be spoken with the Riva voice. Without a key, or with `tts_engine: qwen3`, the voice runs on this computer.
- Speech recognition, embeddings and the fallback model always run on this computer.
- The microphone is only open while `stt_node` runs and is released when you press `Ctrl+C`, close the terminal, or the node crashes. Speech outside a conversation is dropped immediately and is not logged, stored or sent anywhere. **During a conversation the robot answers everything it hears, including other people in the room**, so end it with "goodbye" when you are done, or set `conversation_idle_timeout_s`.
- Long-term memory stores your conversations in `~/.local/share/robot_voice_assistant/memory.jsonl` on this computer. Turn it off with `use_memory: false`, or delete it with `/voice/forget`.
