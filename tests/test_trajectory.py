import json
import unittest
import zipfile

from bg_rl.mat import parse_match_text
from bg_rl.trajectory import action_to_key, match_to_records, record_to_json


class TrajectoryTests(unittest.TestCase):
    def test_export_record_has_selected_action_index(self) -> None:
        with zipfile.ZipFile("data/Arkadium_Backgammon_full_data_gamelogs_001.zip") as zf:
            text = zf.read("1200001.mat").decode("utf-8")
        parsed = parse_match_text(text)

        record = match_to_records(parsed, source_file="1200001.mat")[0]

        selected_index = record["selected_action_index"]
        self.assertGreaterEqual(selected_index, 0)
        self.assertEqual(
            record["legal_action_keys"][selected_index],
            record["selected_action_key"],
        )
        self.assertTrue(record["selected_action_is_full_legal_action"])
        self.assertEqual(record["state"]["turn"], 0)
        self.assertEqual(record["source_file"], "1200001.mat")
        self.assertEqual(record["game_outcome"]["winner"], 0)
        self.assertEqual(record["result_from_player_perspective"], 6)

    def test_record_serializes_as_json(self) -> None:
        with zipfile.ZipFile("data/Arkadium_Backgammon_full_data_gamelogs_001.zip") as zf:
            text = zf.read("1200002.mat").decode("utf-8")
        parsed = parse_match_text(text)

        encoded = record_to_json(match_to_records(parsed, source_file="1200002.mat")[0])
        decoded = json.loads(encoded)

        self.assertIn("legal_actions", decoded)
        self.assertEqual(decoded["selected_action_key"], "11/13:2 0/1:1")
        self.assertEqual(decoded["result_from_player_perspective"], -2)

    def test_action_key_for_pass_action_is_empty(self) -> None:
        self.assertEqual(action_to_key(()), "")


if __name__ == "__main__":
    unittest.main()
