"""The live stall demo: visitor after visitor comes to the robot's stall with something in their hand.

Each round: the robot is put back at its place, a visitor walks up holding a random object (a different one from
the last visitor's) and holds it out; the robot greets them ("Hello, may I help you?") and listens without the
wake word, so you can talk to it through the microphone: ask what the visitor is holding, what it sees, anything
else, or ask it to wave, walk or turn (it keeps a safe distance from people). When you say goodbye, the visitor
leaves and the next round starts.

Needs the scene and the assistant running; start_stall_demo.sh starts all three.
"""

from __future__ import annotations

import random
import time

from stall_demo import OBJECTS

GOODBYE_S = 5.0  # after the conversation ends, the robot says goodbye before the visitor turns to go
PAUSE_S = 3.0  # between one visitor leaving and the next round


def pick_object(previous: str | None, rng: random.Random) -> str:
    """A random object, never the same as the last visitor's."""
    return rng.choice([name for name in OBJECTS if name != previous])


class StallRounds:
    """Decides what happens next from the visitor's state (/gitex/visit_state) and whether a conversation is on
    (/voice/conversation). ``update`` returns ("new_round", object), ("leave",) or None."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self.round = 0
        self.held: str | None = None
        self._phase, self._since = "new", -PAUSE_S

    def _go(self, phase: str, now: float) -> None:
        self._phase, self._since = phase, now

    def update(self, now: float, state: str, in_conversation: bool):
        phase = self._phase
        if phase == "new":
            if state == "idle" and now - self._since >= PAUSE_S:
                self.round += 1
                self.held = pick_object(self.held, self._rng)
                self._go("arriving", now)
                return ("new_round", self.held)
        elif phase in ("arriving", "at_stall", "talking") and state == "idle" and now - self._since > 1.0:
            self._go("new", now)  # they left by themselves (their longest stay is over)
        elif phase == "arriving" and state.startswith("at_stall"):
            self._go("at_stall", now)
        elif phase == "at_stall" and in_conversation:
            self._go("talking", now)
        elif phase == "talking" and not in_conversation:
            self._go("goodbye", now)
        elif phase == "goodbye" and now - self._since >= GOODBYE_S:
            self._go("leaving", now)
            return ("leave",)
        elif phase == "leaving" and state == "idle":
            self._go("new", now)
        return None


def run(seed: int | None) -> None:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Bool, String

    class Host(Node):
        def __init__(self) -> None:
            super().__init__("live_stall")
            self.visit_pub = self.create_publisher(String, "/gitex/visit", 10)
            self.reset_pub = self.create_publisher(String, "/voice/reset", 10)
            self.robot_reset_pub = self.create_publisher(String, "/scene/reset_robot", 10)
            self.state, self.in_conversation = "", False
            self.create_subscription(String, "/gitex/visit_state", lambda m: setattr(self, "state", m.data), 10)
            self.create_subscription(Bool, "/voice/conversation", lambda m: setattr(self, "in_conversation", m.data),
                                     10)

    rclpy.init()
    host = Host()
    print("Waiting for the scene (/gitex/visit_state) and the assistant...", flush=True)
    while rclpy.ok() and (host.state == "" or host.count_publishers("/voice/conversation") == 0):
        rclpy.spin_once(host, timeout_sec=0.2)
    rounds = StallRounds(random.Random(seed))
    try:
        while rclpy.ok():
            rclpy.spin_once(host, timeout_sec=0.2)
            action = rounds.update(time.monotonic(), host.state, host.in_conversation)
            if action is None:
                continue
            if action[0] == "new_round":
                host.robot_reset_pub.publish(String(data="reset"))
                host.reset_pub.publish(String(data="new visitor"))  # a new person: the old conversation is over
                host.visit_pub.publish(String(data=action[1]))
                print(f"\nRound {rounds.round}: a visitor walks up to the stall holding {action[1]}. "
                      "Talk to the robot when it greets them; say goodbye to end the visit.", flush=True)
            elif action[0] == "leave":
                host.visit_pub.publish(String(data="leave"))
                print(f"Round {rounds.round}: goodbye said; the visitor leaves.", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        host.destroy_node()
        if rclpy.ok():  # Ctrl+C may already have shut ROS down
            rclpy.shutdown()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=None, help="repeat the same objects (default: new ones each time)")
    run(parser.parse_args().seed)


if __name__ == "__main__":
    main()
