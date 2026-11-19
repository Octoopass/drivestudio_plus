import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from drivestudio_run import build_command  # noqa: E402


class BuildCommandTests(unittest.TestCase):
    def test_overrides_come_last_and_only_set_values_are_passed(self):
        cmd = build_command({
            "DRIVESTUDIO_CONFIG": "configs/streetgs_ut.yaml",
            "DRIVESTUDIO_OUT": "./o",
            "DRIVESTUDIO_RUN_NAME": "r1",
            "DRIVESTUDIO_DATASET": "nuscenes/1cams",
            "DRIVESTUDIO_SCENE_IDX": "2",
            "DRIVESTUDIO_FASTGS": "1",
            "DRIVESTUDIO_OVERRIDES": "data.pixel_source.load_smpl=False a.b=\"x y\"",
        }, python="py")
        self.assertEqual(cmd[:3], ["py", "tools/train.py", "--config_file"])
        first_override = cmd.index("dataset=nuscenes/1cams")
        self.assertTrue(all("=" in c for c in cmd[first_override:]), "overrides must be last (argparse REMAINDER)")
        self.assertIn("data.scene_idx=2", cmd)
        self.assertIn("trainer.fastgs.enabled=true", cmd)
        self.assertIn("a.b=x y", cmd)
        self.assertFalse(any(c.startswith("trainer.render") or c.startswith("trainer.optim") for c in cmd))

    def test_config_is_required(self):
        with self.assertRaises(ValueError):
            build_command({})

    def test_viewer_and_resume(self):
        cmd = build_command({"DRIVESTUDIO_CONFIG": "c.yaml", "DRIVESTUDIO_VIEWER": "1", "DRIVESTUDIO_RESUME": "ck.pth"}, "py")
        self.assertIn("--enable_viewer", cmd)
        self.assertEqual(cmd[cmd.index("--resume_from") + 1], "ck.pth")


if __name__ == "__main__":
    unittest.main()
