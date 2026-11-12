"""End-to-end test: a visitor walks up to the robot's GITEX stall and has a whole conversation with it.

The visitor (holding an apple) is greeted, then talks to the robot turn by turn: small talk, a question from the
knowledge base, what is in their hand and what the robot sees (YOLO), gestures, and walking and turning. Every turn is
checked on what really happened: the reply, the gesture the robot played, and how far the robot walked and turned
on its odometry. Tokens, timings and Groq's limits are recorded for every answer.

Needs the scene and the assistant running (the microphone is not used; the visitor's words go on /voice/transcript):
    bash start_isaac.sh --gitex --view booth --headless
    bash start_assistant.sh use_microphone:=false config:=<a config with a scratch memory_file>
    source /opt/ros/jazzy/setup.bash && python3 training/visitor_conversation.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import time
from dataclasses import dataclass, field

from stall_demo import GREETING, score_answer

HELD = "apple"


@dataclass(frozen=True)
class Turn:
    say: str
    motion: str | None = None  # the motion the robot must choose (and do, measured on the odometry)
    gestures: set[str] = field(default_factory=set)  # one of these must be played
    says: list[str] = field(default_factory=list)  # the reply must mention one of these words
    holds: str | None = None  # the reply must name this, and nothing else the visitors carry
    about_the_view: bool = False  # the reply must describe the people around
    walk_s: float = 0.0  # keep_walking: how long (simulated seconds) to let the robot walk before the next turn


TURNS = [
    Turn("Hi! What's your name?", says=["R1"]),
    Turn("Can you tell me what is in my hand?", holds=HELD),
    Turn("What can you see around you right now?", about_the_view=True),
    Turn("Can you wave at me?", gestures={"wave_right", "wave_left"}),
    Turn("Please turn left.", motion="turn_left"),
    Turn("Now turn right.", motion="turn_right"),
    Turn("Come here.", motion="walk_forward"),
    Turn("Step back a little.", motion="walk_backward"),
    Turn("Turn around.", motion="turn_around"),
    Turn("Keep walking.", motion="keep_walking", walk_s=3.0),
    Turn("Stop.", motion="stop"),
    Turn("Turn around again so you face me.", motion="turn_around"),
    Turn("Can you celebrate for me?", gestures={"celebrate"}),
    Turn("What does Aideology do?", says=["AI"]),
    Turn("Can you bow?", gestures={"bow"}),
    Turn("Thank you, goodbye!", gestures={"wave_right", "wave_left", "bow"}),
]

# (forward metres, turned degrees) a motion must reach: (low, high), None = not checked
MOTION_RANGES: dict[str, tuple[tuple[float, float] | None, tuple[float, float] | None]] = {
    "turn_left": (None, (60.0, 120.0)),
    "turn_right": (None, (-120.0, -60.0)),
    "walk_forward": ((0.6, 1.4), None),
    "walk_backward": ((-0.8, -0.25), None),
    "keep_walking": ((0.3, 10.0), None),
    "stop": ((-0.15, 0.15), None),  # measured over 2 s after stopping: standing still
}
TURN_AROUND_MIN_DEG = 150.0
MAX_STOP_S = 3.0  # from "Stop." to the robot no longer walking
VIEW_WORDS = ["people", "person", "visitor", "someone", "crowd"]


def motion_delta(before: tuple[float, float, float], after: tuple[float, float, float]) -> tuple[float, float]:
    """How far the robot moved forward (metres, in its heading at ``before``) and turned (degrees, left positive)
    between two odometry poses (x, y, yaw)."""
    dx, dy = after[0] - before[0], after[1] - before[1]
    forward = dx * math.cos(before[2]) + dy * math.sin(before[2])
    turned = math.degrees(math.atan2(math.sin(after[2] - before[2]), math.cos(after[2] - before[2])))
    return forward, turned


def _mentions(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word.lower())}\b", text.lower()) is not None


def check_turn(turn: Turn, reply: str | None, gestures: list[str], motions: list[str],
               moved: tuple[float, float] | None) -> list[str]:
    """What went wrong in one turn; empty when the robot did what the visitor asked."""
    if reply is None:
        return ["no reply"]
    problems = []
    if turn.says and not any(_mentions(reply, word) for word in turn.says):
        problems.append(f"the reply does not mention {' or '.join(turn.says)}")
    if turn.holds and score_answer(reply, turn.holds) != "right":
        problems.append(f"the reply does not name the {turn.holds}")
    if turn.about_the_view and not any(_mentions(reply, word) for word in VIEW_WORDS):
        problems.append("the reply does not describe the people around")
    if turn.gestures and not set(gestures) & turn.gestures:
        problems.append(f"gesture {', '.join(gestures) or 'none'}, not {' or '.join(sorted(turn.gestures))}")
    if turn.motion:
        if turn.motion not in motions:
            problems.append(f"the robot chose {', '.join(motions) or 'no motion'}, not {turn.motion}")
        if moved is None:
            if turn.motion in motions:
                problems.append("the motion was not measured on the odometry")
        else:
            forward, turned = moved
            if turn.motion == "turn_around":
                if abs(turned) < TURN_AROUND_MIN_DEG:
                    problems.append(f"turned {turned:.0f} degrees, not about 180")
            else:
                metres, degrees = MOTION_RANGES[turn.motion]
                if metres and not metres[0] <= forward <= metres[1]:
                    problems.append(f"moved {forward:.2f} m forward, expected {metres[0]} to {metres[1]}")
                if degrees and not degrees[0] <= turned <= degrees[1]:
                    problems.append(f"turned {turned:.0f} degrees, expected {degrees[0]:.0f} to {degrees[1]:.0f}")
    return problems


def capacity(requests: int, tokens: int, turns: int, minutes: float, rpd: int, tpd: int, rpm: int,
             tpm: int) -> dict:
    """How far Groq's limits go at this conversation's use: conversations a day (the tighter of requests and tokens
    a day) and answers a minute at the token-per-minute limit."""
    by_requests, by_tokens = rpd / max(1, requests), tpd / max(1, tokens)
    tokens_per_turn = tokens / max(1, turns)
    return {
        "conversations_per_day": min(by_requests, by_tokens),
        "conversations_per_day_by_requests": by_requests,
        "conversations_per_day_by_tokens": by_tokens,
        "limited_by": "tokens per day" if by_tokens < by_requests else "requests per day",
        "tokens_per_turn": tokens_per_turn,
        "turns_per_minute_at_tpm": tpm / tokens_per_turn if tokens_per_turn else float("inf"),
        "turns_per_minute_at_rpm": rpm,
        "turns_per_minute_in_this_test": turns / minutes if minutes else 0.0,
    }


def _gpu() -> tuple[int, int] | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        temperature, clock = out.strip().split(", ")
        return int(temperature), int(clock)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def run(out_dir: str) -> dict:
    import numpy as np
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import Image
    from std_msgs.msg import Bool, String

    class Visitor(Node):
        def __init__(self) -> None:
            super().__init__("visitor_conversation")
            self.visit_pub = self.create_publisher(String, "/gitex/visit", 10)
            self.say_pub = self.create_publisher(String, "/voice/transcript", 10)
            self.reset_pub = self.create_publisher(String, "/voice/reset", 10)
            self.state, self.scene, self.pose, self.image, self.speaking = "", "", None, None, False
            self.sim_time: float | None = None
            self.log: dict[str, list] = {name: [] for name in
                                         ("reply", "gesture", "played", "motion", "active", "usage", "speaking",
                                          "greet", "scene", "sim")}
            self._sub(String, "/gitex/visit_state", lambda m: setattr(self, "state", m.data))
            self._sub(String, "/vision/scene", self._on_scene)
            self._sub(String, "/voice/reply", lambda m: self._add("reply", m.data))
            self._sub(String, "/robot/gesture", lambda m: self._add("gesture", m.data))
            self._sub(String, "/robot/active_gesture", lambda m: self._add("played", m.data))
            self._sub(String, "/robot/motion", lambda m: self._add("motion", m.data))
            self._sub(String, "/robot/active_motion", lambda m: self._add("active", m.data))
            self._sub(String, "/voice/usage", lambda m: self._add("usage", json.loads(m.data)))
            self._sub(String, "/vision/person_arrived", lambda m: self._add("greet", m.data))
            self._sub(Bool, "/voice/speaking", self._on_speaking)
            self._sub(Odometry, "/robot/odom", self._on_odom)
            self._sub(Clock, "/clock", self._on_clock)
            self.create_subscription(Image, "/robot/camera/image_raw", self._on_image, 2)

        def _sub(self, kind, topic, callback) -> None:
            self.create_subscription(kind, topic, callback, 10)

        def _add(self, name: str, value) -> None:
            self.log[name].append((time.monotonic(), value))

        def _on_scene(self, msg) -> None:
            self.scene = msg.data
            self._add("scene", msg.data)

        def _on_speaking(self, msg) -> None:
            if msg.data != self.speaking:
                self._add("speaking", msg.data)
            self.speaking = msg.data

        def _on_odom(self, msg) -> None:
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            self.pose = (p.x, p.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))

        def _on_clock(self, msg) -> None:
            self.sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9
            if not self.log["sim"] or time.monotonic() - self.log["sim"][-1][0] >= 1.0:
                self._add("sim", self.sim_time)

        def _on_image(self, msg) -> None:
            self.image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)[..., :3]

        def since(self, name: str, moment: float) -> list:
            return [value for at, value in self.log[name] if at >= moment]

        def wait(self, seconds: float, until=None) -> bool:
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                rclpy.spin_once(self, timeout_sec=0.05)
                if until is not None and until():
                    return True
            return until is None

        def wait_quiet(self, timeout: float = 30.0) -> None:
            self.wait(1.0)
            self.wait(timeout, lambda: not self.speaking)

        def save(self, name: str) -> str | None:
            if self.image is None:
                return None
            from PIL import Image as PilImage

            path = os.path.join(out_dir, f"{name}.png")
            PilImage.fromarray(np.ascontiguousarray(self.image)).save(path)
            return path

    rclpy.init()
    visitor = Visitor()
    os.makedirs(out_dir, exist_ok=True)
    print("Waiting for the scene, the robot's odometry and the assistant...", flush=True)
    visitor.wait(180.0, lambda: visitor.state != "" and visitor.pose is not None)
    if visitor.state == "" or visitor.pose is None:
        raise SystemExit("The GITEX scene is not running: start start_isaac.sh --gitex first.")
    topics = ("/voice/reply", "/vision/person_arrived", "/voice/usage", "/robot/active_motion")
    visitor.wait(60.0, lambda: all(visitor.count_publishers(topic) > 0 for topic in topics))
    visitor.wait(60.0, lambda: visitor.state == "idle")
    visitor.wait(2.0)
    gpu_samples = [g for g in [_gpu()] if g]
    started = time.monotonic()

    visitor.reset_pub.publish(String(data="new visitor"))
    visitor.visit_pub.publish(String(data=HELD))
    print(f"\nA visitor walks up to the stall holding an {HELD}...", flush=True)
    arrived = visitor.wait(180.0, lambda: visitor.state == f"at_stall {HELD}")
    arrived_at = time.monotonic()
    greeted = visitor.wait(30.0, lambda: GREETING in visitor.since("reply", started))
    greeting_s = time.monotonic() - arrived_at if greeted else None
    print(f"Robot: {GREETING if greeted else '(no greeting)'}", flush=True)
    visitor.wait_quiet()
    visitor.save("00_greeting")

    results = []
    for number, turn in enumerate(TURNS, start=1):
        before, asked = visitor.pose, time.monotonic()
        print(f"\n[{number:2d}] Visitor: {turn.say}", flush=True)
        visitor.say_pub.publish(String(data=turn.say))
        answered = visitor.wait(60.0, lambda: any(text != GREETING for text in visitor.since("reply", asked)))
        replied_at = time.monotonic()
        reply = next((text for text in visitor.since("reply", asked) if text != GREETING), None)
        print(f"     Robot: {reply if answered else '(no reply)'}", flush=True)
        moved, stop_s = None, None
        if turn.motion:  # the motion command can arrive just after the reply
            visitor.wait(2.0, lambda: turn.motion in visitor.since("motion", asked))
        if turn.motion and turn.motion in visitor.since("motion", asked):
            if turn.motion == "keep_walking":
                visitor.wait(20.0, lambda: "keep_walking" in visitor.since("active", asked))
                walked_from = visitor.sim_time  # simulated seconds: the scene runs at about a fifth of real time
                visitor.wait(120.0, lambda: visitor.sim_time - walked_from >= turn.walk_s)
                moved = motion_delta(before, visitor.pose)
            elif turn.motion == "stop":
                visitor.wait(20.0, lambda: "none" in visitor.since("active", asked))
                stop_s = time.monotonic() - asked
                visitor.wait(1.0)
                still = visitor.pose
                visitor.wait(2.0)
                moved = motion_delta(still, visitor.pose)
            else:
                visitor.wait(20.0, lambda: turn.motion in visitor.since("active", asked))
                visitor.wait(180.0, lambda: "none" in visitor.since("active", asked))
                visitor.wait(1.0)  # let the robot settle
                moved = motion_delta(before, visitor.pose)
        if turn.motion != "keep_walking":
            visitor.wait_quiet()
        image = visitor.save(f"{number:02d}") if turn.holds or turn.about_the_view else None
        gestures = visitor.since("gesture", asked)
        problems = check_turn(turn, reply if answered else None, gestures, visitor.since("motion", asked), moved)
        if stop_s is not None and stop_s > MAX_STOP_S:
            problems.append(f"took {stop_s:.1f} s to stop")
        usage = next(iter(visitor.since("usage", asked)), {})
        voice = [at for at, on in visitor.log["speaking"] if on and at >= replied_at - 0.1]
        results.append({
            "turn": number, "said": turn.say, "reply": reply, "ok": not problems, "problems": problems,
            "gestures": gestures, "played": visitor.since("played", asked),
            "motions": visitor.since("motion", asked),
            "moved_m": None if moved is None else round(moved[0], 2),
            "turned_deg": None if moved is None else round(moved[1], 1),
            "reply_s": round(replied_at - asked, 2) if answered else None,
            "stop_s": None if stop_s is None else round(stop_s, 2),
            "voice_after_reply_s": round(voice[0] - replied_at, 2) if voice else None,
            "usage": usage, "camera_saw": visitor.scene, "image": image,
        })
        gpu_samples += [g for g in [_gpu()] if g]
        mark = "PASS" if not problems else "FAIL: " + "; ".join(problems)
        extra = "" if moved is None else f" (moved {moved[0]:.2f} m, turned {moved[1]:.0f} deg)"
        print(f"     {mark}{extra}", flush=True)
        write_report(results, out_dir, {})  # after every turn: a run stopped half-way keeps what it found

    visitor.visit_pub.publish(String(data="leave"))
    visitor.wait(5.0)
    finished = time.monotonic()
    sims = visitor.log["sim"]
    in_window = [(at, sim) for at, sim in sims if started <= at <= finished]
    sim_speed = ((in_window[-1][1] - in_window[0][1]) / (in_window[-1][0] - in_window[0][0])
                 if len(in_window) > 1 and in_window[-1][0] > in_window[0][0] else None)
    scenes = [at for at, _ in visitor.log["scene"] if started <= at <= finished]
    run_info = {
        "arrived": arrived, "greeted": greeted, "greeting_s": None if greeting_s is None else round(greeting_s, 2),
        "greetings_during_conversation": len([1 for at, text in visitor.log["reply"]
                                              if text == GREETING and at > arrived_at + (greeting_s or 0) + 1]),
        "minutes": round((finished - started) / 60, 2),
        "sim_speed": None if sim_speed is None else round(sim_speed, 2),
        "vision_hz": round(len(scenes) / (finished - started), 2) if finished > started else None,
        "gpu_max_c": max((t for t, _ in gpu_samples), default=None),
        "gpu_min_clock_mhz": min((c for _, c in gpu_samples), default=None),
    }
    visitor.destroy_node()
    rclpy.shutdown()
    write_report(results, out_dir, run_info)
    return {"turns": results, "run": run_info}


def summarise(results: list[dict], run_info: dict) -> dict:
    llm = [r["usage"] for r in results if r["usage"].get("backend") in ("groq", "llama")]
    tokens = [u.get("prompt_tokens", 0) + u.get("completion_tokens", 0) for u in llm]
    limits = next((u for u in reversed(llm) if u.get("requests_per_day")), {})
    summary = {
        "turns": len(results), "passed": sum(r["ok"] for r in results),
        "backends": {name: sum(1 for r in results if r["usage"].get("backend") == name)
                     for name in ("groq", "llama", "commands")},
        "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in llm),
        "completion_tokens": sum(u.get("completion_tokens", 0) for u in llm),
        "tokens_first_answer": tokens[0] if tokens else None, "tokens_last_answer": tokens[-1] if tokens else None,
        "reply_s_median": statistics.median([r["reply_s"] for r in results if r["reply_s"] is not None] or [0]),
        "reply_s_max": max([r["reply_s"] for r in results if r["reply_s"] is not None] or [0]),
        "voice_after_reply_s_median": statistics.median(
            [r["voice_after_reply_s"] for r in results if r["voice_after_reply_s"] is not None] or [0]),
        "groq_limits": {key: limits.get(key) for key in ("requests_per_day", "requests_left_today",
                                                         "tokens_per_minute", "tokens_left_this_minute")},
    }
    steps: dict[str, list[float]] = {}
    for u in llm:
        for name, seconds in u.get("steps_s", {}).items():
            steps.setdefault(name, []).append(seconds)
    summary["steps_s_median"] = {name: round(statistics.median(values), 2) for name, values in steps.items()}
    if llm and run_info.get("minutes"):
        summary["capacity"] = capacity(len(llm), sum(tokens), len(llm), run_info["minutes"],
                                       rpd=limits.get("requests_per_day") or GROQ_LIMITS["rpd"], tpd=GROQ_LIMITS["tpd"],
                                       rpm=GROQ_LIMITS["rpm"], tpm=limits.get("tokens_per_minute") or GROQ_LIMITS["tpm"])
    return summary


# Groq's published base limits for qwen/qwen3.8-27b (console.groq.com/docs/rate-limits, read 2026-11-11); the
# response headers give this key's own requests per day and tokens per minute, which take precedence.
GROQ_LIMITS = {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200_000}


def write_report(results: list[dict], out_dir: str, run_info: dict) -> str:
    summary = summarise(results, run_info)
    with open(os.path.join(out_dir, "conversation.json"), "w") as f:
        json.dump({"run": run_info, "summary": summary, "turns": results}, f, indent=2)
    lines = ["# Visitor conversation: end-to-end test", "",
             f"**{summary['passed']} of {summary['turns']} turns passed.** "
             f"Greeted: {run_info.get('greeted', '?')} ({run_info.get('greeting_s', '?')} s after arriving).", "",
             "| # | Visitor | Robot | Gesture | Motion | Moved | Reply | Tokens | Result |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        moved = "" if r["moved_m"] is None else f"{r['moved_m']} m, {r['turned_deg']}°"
        tokens = (r["usage"].get("prompt_tokens", 0) + r["usage"].get("completion_tokens", 0)) if r["usage"] else ""
        result = "pass" if r["ok"] else "FAIL: " + "; ".join(r["problems"])
        lines.append(f"| {r['turn']} | {r['said']} | {r['reply']} | {', '.join(r['gestures'])} | "
                     f"{', '.join(r['motions'])} | {moved} | {r['reply_s']} s | {tokens} | {result} |")
    path = os.path.join(out_dir, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", default=os.path.expanduser(
        f"~/g1_voice_ws/log/visitor_conversation/{time.strftime('%Y%m%d_%H%M%S')}"))
    args = parser.parse_args()
    outcome = run(args.out_dir)
    summary = summarise(outcome["turns"], outcome["run"])
    print(f"\nRESULT: {summary['passed']}/{summary['turns']} turns passed; report: "
          f"{os.path.join(args.out_dir, 'report.md')}", flush=True)
    print(json.dumps({"run": outcome["run"], "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
