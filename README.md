# PhotSat Calibration Pipelines

This repository bundles several Python modules used to calibrate the PhotSat imaging sensors:

- **bias_pipeline** – temperature dependent electronic bias model.
- **dark_pipeline** – dark current model for long exposures.
- **flat_pipeline** – flat field generation utilities.

## Installation

Install the required Python packages using the repository wide `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Each pipeline can then be executed through its own `run_pipeline.py` script. See the individual module `README.md` files for detailed usage instructions.

