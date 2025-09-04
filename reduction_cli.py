"""Interactive CLI for data reduction pipeline.

This module provides a simple text-based interface to configure and
run a three-step reduction pipeline.  Each step expects an input
directory containing FITS files and an output directory where processed
files will be copied.  The CLI displays the state of each step in a
compact table and allows the user to configure directories and start
the processing.

Usage:
    python reduction_cli.py

Within the CLI use:
    config STEP INPUT_DIR OUTPUT_DIR  # configure a pipeline step
    run                               # execute the pipeline
    status                            # show pipeline status
    exit                              # quit the application
"""

from __future__ import annotations

import cmd
import os
import shutil
from dataclasses import dataclass, field
from glob import glob
from typing import Callable, List, Optional


# ---------------------------------------------------------------------------
# Pipeline core
# ---------------------------------------------------------------------------
@dataclass
class PipelineStep:
    """Represents a processing step in the pipeline."""

    name: str
    processor: Callable[[str, str], None]
    input_dir: Optional[str] = None
    output_dir: Optional[str] = None
    status: str = "PENDING"
    processed: int = 0
    total: int = 0

    def run(self) -> None:
        if self.input_dir is None or self.output_dir is None:
            raise RuntimeError(f"Step '{self.name}' directories not configured")
        fits_files = glob(os.path.join(self.input_dir, "*.fits"))
        self.total = len(fits_files)
        self.processed = 0
        self.status = "RUNNING"
        os.makedirs(self.output_dir, exist_ok=True)
        for f in fits_files:
            self.processor(f, self.output_dir)
            self.processed += 1
        self.status = "DONE"


@dataclass
class DataPipeline:
    """Sequential pipeline composed of multiple :class:`PipelineStep`"""

    steps: List[PipelineStep] = field(default_factory=list)

    def configure(self, step_name: str, input_dir: str, output_dir: str) -> None:
        for step in self.steps:
            if step.name == step_name:
                step.input_dir = input_dir
                step.output_dir = output_dir
                return
        raise ValueError(f"Unknown step '{step_name}'")

    def run(self) -> None:
        for step in self.steps:
            step.run()

    def display_status(self) -> None:
        header = f"{'Step':10}{'Status':12}{'Progress':10}"
        print(header)
        print("-" * len(header))
        for step in self.steps:
            progress = f"{step.processed}/{step.total}"
            print(f"{step.name:10}{step.status:12}{progress:10}")


# ---------------------------------------------------------------------------
# Processing functions
# ---------------------------------------------------------------------------
def copy_processor(file_path: str, output_dir: str) -> None:
    """Simple processor that copies the FITS file to the output directory."""
    shutil.copy(file_path, os.path.join(output_dir, os.path.basename(file_path)))


def build_default_pipeline() -> DataPipeline:
    """Create a pipeline with the default sdp1/sdp2/sdp3 steps."""
    steps = [
        PipelineStep("sdp1", copy_processor),
        PipelineStep("sdp2", copy_processor),
        PipelineStep("sdp3", copy_processor),
    ]
    return DataPipeline(steps)


# ---------------------------------------------------------------------------
# Interactive shell
# ---------------------------------------------------------------------------
class PipelineShell(cmd.Cmd):
    intro = "Data Reduction Pipeline CLI. Type 'help' for commands."
    prompt = "pipeline> "

    def __init__(self, pipeline: DataPipeline):
        super().__init__()
        self.pipeline = pipeline

    # ----- basic commands -----
    def do_config(self, arg: str) -> None:
        """config STEP INPUT_DIR OUTPUT_DIR"""
        parts = arg.split()
        if len(parts) != 3:
            print("Usage: config STEP INPUT_DIR OUTPUT_DIR")
            return
        step, in_dir, out_dir = parts
        try:
            self.pipeline.configure(step, in_dir, out_dir)
        except ValueError as exc:
            print(exc)
        else:
            print(f"Configured {step}: in={in_dir}, out={out_dir}")

    def do_run(self, arg: str) -> None:
        """Run the pipeline"""
        try:
            self.pipeline.run()
        except Exception as exc:  # pragma: no cover - simple CLI error display
            print(f"Error: {exc}")
        else:
            self.pipeline.display_status()

    def do_status(self, arg: str) -> None:
        """Show pipeline status"""
        self.pipeline.display_status()

    def do_exit(self, arg: str) -> bool:  # pragma: no cover - CLI exit
        """Exit the CLI"""
        print("Bye")
        return True

    do_quit = do_exit


def main() -> None:
    pipeline = build_default_pipeline()
    pipeline.display_status()
    PipelineShell(pipeline).cmdloop()


if __name__ == "__main__":
    main()
