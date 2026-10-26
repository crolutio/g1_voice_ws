"""Shared entry point for the package's nodes, with clean shutdown when the terminal closes."""

from __future__ import annotations

import signal
from collections.abc import Callable

import rclpy
from rclpy.node import Node


def run_node(create_node: Callable[[], Node]) -> None:
    rclpy.init()
    # Closing the terminal sends SIGHUP; treat it like Ctrl-C so the microphone is released.
    signal.signal(signal.SIGHUP, lambda *_: rclpy.try_shutdown())
    node = create_node()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
