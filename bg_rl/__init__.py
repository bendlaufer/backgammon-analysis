"""Backgammon RL primitives.

This package is intentionally environment-first: exact state transitions,
legal actions, and tensor encodings that can support self-play training.
"""

from bg_rl.state import BackgammonState, CubeState, MoveStep

__all__ = ["BackgammonState", "CubeState", "MoveStep"]
