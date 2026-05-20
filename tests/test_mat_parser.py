import unittest
import zipfile

from bg_rl.mat import parse_match_text
from bg_rl.state import MoveStep


class MatParserTests(unittest.TestCase):
    def test_parse_first_sample_log(self) -> None:
        with zipfile.ZipFile("data/Arkadium_Backgammon_full_data_gamelogs_001.zip") as zf:
            text = zf.read("1200001.mat").decode("utf-8")

        parsed = parse_match_text(text)

        self.assertEqual(parsed.match_length, 3)
        self.assertGreater(len(parsed.decisions), 0)
        self.assertEqual(parsed.game_outcomes[0].winner, 0)
        self.assertEqual(parsed.game_outcomes[0].points_won, 6)
        self.assertEqual(parsed.decisions[0].dice, (3, 5))
        self.assertEqual(
            parsed.decisions[0].action,
            (MoveStep(7, 2, 5, False), MoveStep(5, 2, 3, False)),
        )

    def test_parse_game_where_second_player_opens(self) -> None:
        with zipfile.ZipFile("data/Arkadium_Backgammon_full_data_gamelogs_001.zip") as zf:
            text = zf.read("1200002.mat").decode("utf-8")

        parsed = parse_match_text(text)

        self.assertEqual(parsed.decisions[0].player, 1)
        self.assertEqual(parsed.decisions[0].dice, (1, 2))
        self.assertEqual(
            parsed.decisions[0].action,
            (MoveStep(11, 13, 2, False), MoveStep(0, 1, 1, False)),
        )


if __name__ == "__main__":
    unittest.main()
