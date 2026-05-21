from __future__ import annotations

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
from pathlib import Path
import random
import sys
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bg_rl.engine import BasicEngine, engine_summary, rankings_to_records
from bg_rl.trajectory import (
    action_from_tokens,
    action_to_key,
    action_to_tokens,
    state_from_dict,
    state_to_dict,
)
from bg_rl.state import BackgammonState


ROOT = Path(__file__).resolve().parent


def roll_dice() -> tuple[int, int]:
    return random.randint(1, 6), random.randint(1, 6)


class BackgammonHandler(SimpleHTTPRequestHandler):
    engine = BasicEngine("artifacts/bc-policy-full-local/model.pt")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "static"), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/new":
            self._send_json(
                {
                    "state": state_to_dict(BackgammonState.initial(turn=0)),
                    "engine": engine_summary(self.engine),
                }
            )
            return
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/legal":
                self._send_json(self._legal(payload))
            elif parsed.path == "/api/play":
                self._send_json(self._play(payload))
            elif parsed.path == "/api/engine":
                self._send_json(self._engine(payload))
            else:
                self.send_error(404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def _legal(self, payload: dict) -> dict:
        state = state_from_dict(payload["state"])
        dice = tuple(payload["dice"])
        actions = state.legal_actions(dice)
        rankings = self.engine.rank_moves(state, dice)
        ranking_by_key = {action_to_key(ranking.action): ranking.score for ranking in rankings}
        return {
            "legal_actions": [
                {
                    "action": action_to_tokens(action),
                    "action_key": action_to_key(action),
                    "score": ranking_by_key.get(action_to_key(action)),
                }
                for action in actions
            ],
        }

    def _play(self, payload: dict) -> dict:
        state = state_from_dict(payload["state"])
        action = action_from_tokens(payload["action"])
        dice = tuple(payload["dice"])
        if action not in state.legal_actions(dice):
            raise ValueError("selected action is not legal")
        next_state = state.apply_action(action)
        return {"state": state_to_dict(next_state), "winner": next_state.winner()}

    def _engine(self, payload: dict) -> dict:
        state = state_from_dict(payload["state"])
        if state.winner() is not None:
            return {"state": state_to_dict(state), "winner": state.winner()}
        if state.turn != 1:
            raise ValueError("engine endpoint expects player 1 to be on turn")
        dice = roll_dice()
        rankings = self.engine.rank_moves(state, dice)
        action = rankings[0].action if rankings else ()
        next_state = state.apply_action(action)
        return {
            "dice": list(dice),
            "action": action_to_tokens(action),
            "action_key": action_to_key(action),
            "rankings": rankings_to_records(rankings[:5]),
            "state": state_to_dict(next_state),
            "winner": next_state.winner(),
            "engine": engine_summary(self.engine),
        }

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length).decode("utf-8")
        return json.loads(data)

    def _send_json(self, payload: dict, *, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the browser backgammon player.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), BackgammonHandler)
    print(f"Backgammon player: http://{args.host}:{args.port}")
    print(f"Engine: {engine_summary(BackgammonHandler.engine)}")
    server.serve_forever()


if __name__ == "__main__":
    main()
