"""Launch the R1 voice assistant: speech-to-text, the brain, text-to-speech, gestures and walking."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

PACKAGE = "robot_voice_assistant"
WHISPER_CACHE = os.path.expanduser("~/g1_voice_ws/models/hf")
# Log lines show the local time to the millisecond, e.g. [INFO] [2026-10-27 10:15:02.123] [brain_node]: ...
LOG_FORMAT = "[{severity}] [{date_time_with_ms}] [{name}]: {message}"


def generate_launch_description() -> LaunchDescription:
    default_config = os.path.join(get_package_share_directory(PACKAGE), "config", "voice_assistant.yaml")
    config = LaunchConfiguration("config")
    use_microphone = LaunchConfiguration("use_microphone")

    # Whisper, Qwen3-TTS and Qwen3-Embedding are loaded from the workspace's model folder.
    model_env = {"HF_HOME": os.environ.get("HF_HOME", WHISPER_CACHE)}

    def node(executable: str, **kwargs: object) -> Node:
        return Node(package=PACKAGE, executable=executable, name=executable, output="screen",
                    parameters=[config], additional_env=model_env, **kwargs)

    return LaunchDescription(
        [
            SetEnvironmentVariable(
                "RCUTILS_CONSOLE_OUTPUT_FORMAT", os.environ.get("RCUTILS_CONSOLE_OUTPUT_FORMAT", LOG_FORMAT)
            ),
            DeclareLaunchArgument("config", default_value=default_config),
            DeclareLaunchArgument("use_microphone", default_value="true",
                                  description="false = type messages on /voice/transcript instead of speaking"),
            DeclareLaunchArgument("use_vision", default_value="true",
                                  description="see through the head camera (YOLO-World): greet people, name objects"),
            DeclareLaunchArgument("use_sim_time", default_value="true",
                                  description="time gestures with Isaac Sim's /clock so they match the simulation speed"),
            node("stt_node", condition=IfCondition(use_microphone)),
            node("brain_node"),
            Node(package=PACKAGE, executable="motion_node", name="motion_node", output="screen",
                 parameters=[config, {"use_sim_time": ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)}]),
            node("tts_node"),
            node("vision_node", condition=IfCondition(LaunchConfiguration("use_vision"))),
            Node(package=PACKAGE, executable="gesture_node", name="gesture_node", output="screen",
                 parameters=[config, {"use_sim_time": ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)}]),
        ]
    )
