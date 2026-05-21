import unittest

from bg_rl.trajectory import state_to_dict
from bg_rl.state import BackgammonState
from web.server import BackgammonHandler


class WebApiTests(unittest.TestCase):
    def test_handler_legal_response_shape(self) -> None:
        payload = {
            "state": state_to_dict(BackgammonState.initial(turn=0)),
            "dice": [3, 1],
        }

        response = BackgammonHandler._legal(BackgammonHandler, payload)

        self.assertIn("legal_actions", response)
        self.assertGreater(len(response["legal_actions"]), 0)


if __name__ == "__main__":
    unittest.main()
