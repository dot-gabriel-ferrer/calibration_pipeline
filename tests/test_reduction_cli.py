import os
from pathlib import Path

from reduction_cli import build_default_pipeline


def _create_dummy_fits(directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text("test")


def test_pipeline_process(tmp_path):
    # create input and output directories
    in1 = tmp_path / "sdp1"
    out1 = tmp_path / "out1"
    in2 = tmp_path / "sdp2"
    out2 = tmp_path / "out2"
    in3 = tmp_path / "sdp3"
    out3 = tmp_path / "out3"

    _create_dummy_fits(in1, "a.fits")
    _create_dummy_fits(in2, "b.fits")
    _create_dummy_fits(in3, "c.fits")

    pipeline = build_default_pipeline()
    pipeline.configure("sdp1", str(in1), str(out1))
    pipeline.configure("sdp2", str(in2), str(out2))
    pipeline.configure("sdp3", str(in3), str(out3))

    pipeline.run()

    # Each step should be done and the output files should exist
    for step, out_dir in zip(pipeline.steps, [out1, out2, out3]):
        assert step.status == "DONE"
        assert step.processed == 1
        files = list(out_dir.glob("*.fits"))
        assert len(files) == 1
