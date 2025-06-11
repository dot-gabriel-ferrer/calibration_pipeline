import pathlib
from utils.raw_to_fits import gather_attempts

def test_gather_attempts_handles_nested_dark(tmp_path):
    root = tmp_path / "TestSection2"
    (root / "20Frames" / "T0" / "0.5s" / "attempt0" / "frames").mkdir(parents=True)
    (root / "ContinuousFrames" / "0.2s" / "T5" / "attempt1" / "frames").mkdir(parents=True)

    attempts = set(map(pathlib.Path, gather_attempts(str(root), max_depth=4)))
    assert (root / "20Frames" / "T0" / "0.5s" / "attempt0") in attempts
    assert (root / "ContinuousFrames" / "0.2s" / "T5" / "attempt1") in attempts


def test_gather_attempts_deep_hierarchy(tmp_path):
    root = tmp_path / "TestSection2"
    (root / "20Frames" / "T20" / "T0" / "0.5s" / "attempt0" / "frames").mkdir(
        parents=True
    )

    attempts = set(map(pathlib.Path, gather_attempts(str(root), max_depth=6)))
    assert (
        root / "20Frames" / "T20" / "T0" / "0.5s" / "attempt0"
    ) in attempts


def test_gather_attempts_frames_without_attempt_dir(tmp_path):
    root = tmp_path / "TestSection2"
    attempt = root / "20Frames" / "T20" / "T0" / "0.4s"
    frames = attempt / "frames"
    frames.mkdir(parents=True)
    (attempt / "configFile.txt").write_text("WIDTH: 1\nHEIGHT: 1\nBIT_DEPTH: 16\n")
    (attempt / "temperatureLog.csv").write_text("FrameNum\n0\n")

    attempts = set(map(pathlib.Path, gather_attempts(str(root), max_depth=6)))
    assert attempt in attempts


def test_gather_attempts_radiation_log_completo(tmp_path):
    root = tmp_path / "TestSection2"
    attempt = root / "T0"
    frames = attempt / "frames"
    frames.mkdir(parents=True)
    (attempt / "configFile.txt").write_text("WIDTH: 1\nHEIGHT: 1\nBIT_DEPTH: 16\n")
    (attempt / "radiationLogCompleto.csv").write_text("FrameNum\n0\n")

    attempts = set(map(pathlib.Path, gather_attempts(str(root), max_depth=2)))
    assert attempt in attempts

