import pathlib
from utils.raw_to_fits import gather_attempts

def test_gather_attempts_handles_nested_dark(tmp_path):
    root = tmp_path / "TestSection2"
    (root / "20Frames" / "T0" / "attempt0" / "frames").mkdir(parents=True)
    (root / "ContinuousFrames" / "T5" / "attempt1" / "frames").mkdir(parents=True)

    attempts = set(map(pathlib.Path, gather_attempts(str(root), nested=True)))
    assert (root / "20Frames" / "T0" / "attempt0") in attempts
    assert (root / "ContinuousFrames" / "T5" / "attempt1") in attempts
