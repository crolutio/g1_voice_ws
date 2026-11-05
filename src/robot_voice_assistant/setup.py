from glob import glob

from setuptools import find_packages, setup

PACKAGE_NAME = "robot_voice_assistant"

setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
        (f"share/{PACKAGE_NAME}/launch", glob("launch/*.launch.py")),
        (f"share/{PACKAGE_NAME}/config", glob("config/*.yaml") + glob("config/*.npz")),
        (f"share/{PACKAGE_NAME}/config/knowledge", glob("config/knowledge/*")),
        (f"share/{PACKAGE_NAME}/scripts", glob("scripts/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="radhika",
    maintainer_email="rkhatri@aideology.ai",
    description="Voice assistant for a simulated Unitree R1 in Isaac Sim.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "stt_node = robot_voice_assistant.stt_node:main",
            "brain_node = robot_voice_assistant.brain_node:main",
            "tts_node = robot_voice_assistant.tts_node:main",
            "gesture_node = robot_voice_assistant.gesture_node:main",
            "motion_node = robot_voice_assistant.motion_node:main",
            "vision_node = robot_voice_assistant.vision_node:main",
            "chat = robot_voice_assistant.chat_cli:main",
        ],
    },
)
