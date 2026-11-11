"""The stall demo and test: visitors bring objects to the robot's GITEX stall and ask what they are holding.

For each visit: a visitor walks up holding the object (/gitex/visit), the robot greets them on its own when its
vision sees someone close ("Hello, may I help you?"), the visitor asks "Can you tell me what is in my hand?"
(/voice/transcript, as if spoken), and the robot's answer (/voice/reply) is scored against what they really hold.
Each answer is spoken by the robot; the report lists every visit, what the camera saw, and the result per object.

Needs the scene and the assistant running (the microphone is not used):
    bash start_isaac.sh --gitex --view booth
    bash start_assistant.sh use_microphone:=false config:=<a config with a scratch memory_file>
    source /opt/ros/jazzy/setup.bash && python3 training/stall_demo.py --rounds 2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

OBJECTS = ["apple", "banana", "mug", "bottle", "scissors"]
NAMES = {"apple": ["apple"], "banana": ["banana"], "mug": ["mug", "cup"], "bottle": ["bottle"],
         "scissors": ["scissors"]}
QUESTION = "Can you tell me what is in my hand?"
GREETING = "Hello, may I help you?"


def _mentions(text: str, names: list[str]) -> bool:
    return any(re.search(rf"\b{re.escape(name)}s?\b", text) for name in names)


def score_answer(answer: str, held: str) -> str:
    """ "right" if the answer names what is held and nothing else on the list, "wrong" if it names something else,
    "not seen" if it names none of them."""
    text = answer.lower()
    right = _mentions(text, NAMES[held])
    others = any(_mentions(text, names) for name, names in NAMES.items() if name != held)
    if right and not others:
        return "right"
    return "wrong" if right or others else "not seen"


def summarise(trials: list[dict]) -> dict:
    """Per object and in total: visits, greetings, and right / wrong / not seen answers."""
    summary: dict[str, dict] = {}
    for key in [t["object"] for t in trials] + ["total"]:
        summary.setdefault(key, {"trials": 0, "greeted": 0, "right": 0, "wrong": 0, "not seen": 0})
    for trial in trials:
        for key in (trial["object"], "total"):
            summary[key]["trials"] += 1
            summary[key]["greeted"] += bool(trial["greeted"])
            summary[key][trial["result"]] += 1
    return summary


def load_trials(out_dir: str) -> list[dict]:
    """The visits an earlier session wrote to ``out_dir`` (the test runs in short sessions: the GPU heats up)."""
    path = os.path.join(out_dir, "trials.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)["trials"]


def run(rounds: int, out_dir: str, objects: list[str], first_round: int = 1) -> list[dict]:
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import Bool, String

    class Demo(Node):
        def __init__(self) -> None:
            super().__init__("stall_demo")
            self.visit_pub = self.create_publisher(String, "/gitex/visit", 10)
            self.transcript_pub = self.create_publisher(String, "/voice/transcript", 10)
            self.reset_pub = self.create_publisher(String, "/voice/reset", 10)
            self.state, self.scene, self.image, self.speaking = "", "", None, False
            self.replies: list[tuple[float, str]] = []
            self.greetings: list[float] = []
            self.create_subscription(String, "/gitex/visit_state", lambda m: setattr(self, "state", m.data), 10)
            self.create_subscription(String, "/vision/scene", lambda m: setattr(self, "scene", m.data), 10)
            self.create_subscription(String, "/voice/reply", lambda m: self.replies.append((time.monotonic(), m.data)), 10)
            self.create_subscription(String, "/vision/person_arrived", lambda _m: self.greetings.append(time.monotonic()), 10)
            self.create_subscription(Bool, "/voice/speaking", lambda m: setattr(self, "speaking", m.data), 10)
            self.create_subscription(Image, "/robot/camera/image_raw", self._on_image, 2)

        def _on_image(self, msg) -> None:
            self.image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

        def wait(self, seconds: float, until=None) -> bool:
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                rclpy.spin_once(self, timeout_sec=0.05)
                if until is not None and until():
                    return True
            return until is None

        def wait_quiet(self, timeout: float = 15.0) -> None:
            """Until the robot has finished speaking (the voice starts a moment after the reply)."""
            self.wait(1.0)
            self.wait(timeout, lambda: not self.speaking)

        def save(self, name: str) -> str | None:
            if self.image is None:
                return None
            from PIL import Image as PilImage

            path = os.path.join(out_dir, f"{name}.png")
            PilImage.fromarray(self.image).save(path)
            return path

    rclpy.init()
    demo = Demo()
    os.makedirs(out_dir, exist_ok=True)
    print("Waiting for the scene (/gitex/visit_state) and the camera...", flush=True)
    demo.wait(120.0, lambda: demo.state != "" and demo.image is not None)
    if demo.state == "" or demo.image is None:
        raise SystemExit("The GITEX scene is not running: start start_isaac.sh --gitex first.")
    demo.wait(60.0, lambda: demo.state == "idle")
    # a freshly started assistant: without this, the first greeting can be sent before this node hears it
    demo.wait(30.0, lambda: all(demo.count_publishers(topic) > 0 for topic in ("/voice/reply", "/vision/person_arrived")))
    demo.wait(2.0)
    trials = load_trials(out_dir)
    for round_number in range(first_round, first_round + rounds):
        for held in objects:
            name = f"{round_number}_{held}"
            demo.reset_pub.publish(String(data="new visitor"))  # a new person: no earlier conversation
            sent = time.monotonic()
            demo.visit_pub.publish(String(data=held))
            print(f"\n[visit {name}] A visitor walks up to the stall holding {held}...", flush=True)
            if not demo.wait(120.0, lambda: demo.state == f"at_stall {held}"):
                print(f"[visit {name}] The visitor never arrived ({demo.state!r}); skipped.", flush=True)
                demo.visit_pub.publish(String(data="leave"))
                demo.wait(60.0, lambda: demo.state == "idle")
                continue
            # the robot greets by itself once its vision sees someone close (on the way in or standing there)
            greeted = demo.wait(20.0, lambda: any(at >= sent for at in demo.greetings))
            greeting = next((text for at, text in demo.replies if at >= sent and text == GREETING), None)
            print(f"[visit {name}] Robot: {greeting or '(no greeting)'}", flush=True)
            demo.wait_quiet()
            scene = demo.scene
            image = demo.save(name)
            asked = time.monotonic()
            print(f"[visit {name}] Visitor: {QUESTION}", flush=True)
            demo.transcript_pub.publish(String(data=QUESTION))
            answered = demo.wait(45.0, lambda: any(at >= asked and text != GREETING for at, text in demo.replies))
            answer_at, answer = next(((at, text) for at, text in demo.replies if at >= asked and text != GREETING),
                                     (time.monotonic(), "(no answer)"))
            result = score_answer(answer, held) if answered else "not seen"
            print(f"[visit {name}] Robot: {answer}", flush=True)
            print(f"[visit {name}] Camera saw: {scene}", flush=True)
            print(f"[visit {name}] Result: {result.upper()}", flush=True)
            demo.wait_quiet()
            trials.append({"visit": name, "object": held, "greeted": greeted, "greeting": greeting, "scene": scene,
                           "question": QUESTION, "answer": answer, "answer_s": round(answer_at - asked, 2),
                           "result": result, "image": image})
            write_report(trials, out_dir)  # after every visit: a run stopped half-way keeps what it found
            demo.visit_pub.publish(String(data="leave"))
            demo.wait(60.0, lambda: demo.state == "idle")
    demo.destroy_node()
    rclpy.shutdown()
    return trials


def write_report(trials: list[dict], out_dir: str) -> str:
    summary = summarise(trials)
    with open(os.path.join(out_dir, "trials.json"), "w") as f:
        json.dump({"trials": trials, "summary": summary}, f, indent=2)
    lines = ["# Stall demo: what is in my hand?", "",
             "| Object | Visits | Greeted | Right | Wrong | Not seen |", "|---|---|---|---|---|---|"]
    for key, row in summary.items():
        label = f"**{key}**" if key == "total" else key
        lines.append(f"| {label} | {row['trials']} | {row['greeted']} | {row['right']} | {row['wrong']} | "
                     f"{row['not seen']} |")
    lines += ["", "| Visit | Robot's answer | Camera saw | Result | Answer time |", "|---|---|---|---|---|"]
    for t in trials:
        lines.append(f"| {t['visit']} | {t['answer']} | {t['scene']} | {t['result']} | {t['answer_s']:.1f} s |")
    path = os.path.join(out_dir, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rounds", type=int, default=2, help="visits per object")
    parser.add_argument("--objects", nargs="+", default=OBJECTS, choices=OBJECTS)
    parser.add_argument("--first-round", type=int, default=1,
                        help="number of the first round; the visits already in --out-dir are kept in the report")
    parser.add_argument("--out-dir", default=os.path.expanduser(f"~/g1_voice_ws/log/stall_demo/{time.strftime('%Y%m%d_%H%M%S')}"))
    args = parser.parse_args()
    trials = run(args.rounds, args.out_dir, args.objects, args.first_round)
    path = write_report(trials, args.out_dir)
    total = summarise(trials).get("total", {})
    print(f"\nRESULT: {total.get('right', 0)}/{total.get('trials', 0)} answers right, "
          f"{total.get('greeted', 0)}/{total.get('trials', 0)} visitors greeted; report: {path}", flush=True)


if __name__ == "__main__":
    main()
