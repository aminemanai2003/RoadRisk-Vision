# Contributing to RoadRisk Vision

Thank you for helping build open, privacy-respecting road-safety research tools.

## Development setup

```powershell
git clone https://github.com/aminemanai2003/RoadRisk-Vision.git
cd RoadRisk-Vision
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,dashboard]"
python -m ruff check src apps tests scripts
python -m pytest
```

Use `agent/<short-description>` or `feature/<short-description>` branches.
Keep commits small, tested and meaningful. Pull requests should explain the
behavioral change, tests and any safety/privacy impact.

## Contribution boundaries

- Never commit real private footage, absolute GPS, credentials, model weights or
  generated `runs/` data.
- New warning rules need evidence, tests and explicit limitations.
- Uncalibrated estimates must remain qualitative.
- Network calls may occur only in explicit setup/download commands, never during
  analysis.
- Schema changes require an update to `docs/data-contract.md` and compatibility
  tests.

By participating, you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
