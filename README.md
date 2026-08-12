# RoadRisk Vision

[![CI](https://github.com/aminemanai2003/RoadRisk-Vision/actions/workflows/ci.yml/badge.svg)](https://github.com/aminemanai2003/RoadRisk-Vision/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)

RoadRisk Vision is a privacy-first, post-drive dashcam analyzer. Record a trip
with a phone, transfer the video to a PC, then create an annotated video and a
versioned dataset of road-risk events. It combines HybridNets road perception,
YOLOX object detection, object tracking, optional camera calibration and
optional GPS telemetry.

![RoadRisk Vision reviewing a completed local analysis](docs/assets/dashboard-review.png)

> [!IMPORTANT]
> RoadRisk Vision is a research and driver-awareness tool. It does not control
> a vehicle, replace an attentive driver, or provide certified collision alerts.

## What it produces

- Silent annotated H.264 video with lanes, objects and advisory warning states.
- `events.jsonl` with reproducible event identifiers and evidence.
- `trip_summary.json` and a flattened `trip_summary.csv`.
- `timeline.json` for the local dashboard.
- `manifest.json` with configuration, model provenance and run status.

## Quick start

Python 3.11 and FFmpeg are required. The default test backend works without
downloading AI weights; real model weights are downloaded explicitly.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,dashboard]"
roadrisk doctor
roadrisk models download
roadrisk analyze trip.mp4 --backend mock
roadrisk dashboard
```

`roadrisk doctor` is read-only. It distinguishes missing and unsupported FFmpeg
installations and prints tested platform-specific setup guidance.

For real inference, install the inference extras and follow
[`docs/models.md`](docs/models.md). RoadRisk Vision never downloads weights
during analysis.

## Architecture

```text
phone video + optional telemetry + optional calibration
                         |
                  FFmpeg normalization
                         |
                 bounded frame stream
                         |
          HybridNets D3 -> YOLOX-S -> tracking
                         |
               geometry and risk engine
                         |
      annotated video + timeline + research exports
```

The GPU stages run sequentially to target GPUs with 4 GB VRAM. Numeric distance
and TTC are omitted unless the camera is calibrated and the estimate is stable.

## Project status

Version `v0.1.0` provides the complete PC post-drive workflow: phone-video
normalization, HybridNets + YOLOX-S perception, two-stage ByteTrack association,
optional calibration and telemetry, annotated video, dashboard review and
versioned research exports. The RTX 2050 release benchmark processed all 300
frames of the licensed 1080p/30 fixture in 4.94x source duration while remaining
inside every memory/performance gate. See the
[full reproducible result](docs/benchmark-results/rtx2050.md) and
[roadmap](docs/roadmap.md).

## Contributing

Contributions are welcome. Start with [`CONTRIBUTING.md`](CONTRIBUTING.md), the
[architecture](docs/architecture.md), and issues labelled `good first issue`.
Never commit private driving footage, GPS traces, credentials, model weights or
generated run directories.

## Upstream and licenses

RoadRisk Vision is a fork of
[datvuthanh/HybridNets](https://github.com/datvuthanh/HybridNets), preserved at
tag `baseline/upstream-8735a699`. HybridNets is MIT licensed. The optional YOLOX
integration is based on [Megvii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
and remains subject to its Apache-2.0 license. See [`NOTICE`](NOTICE).

## Actuarial relevance

The project exports exposure and driving-risk features: analyzed duration,
valid distance, event counts, severity and exposure-normalized rates. It does
not estimate premiums, claim frequency or claim severity. Events are called
`risk_events`; the term *near miss* is reserved for an explicitly defined,
measurable low-TTC event.

Generate a portfolio-style research summary with `roadrisk exposure-report`.
See the [actuarial telematics guide](docs/actuarial-telematics.md) for the exact
boundary between driving features and genuine actuarial claims modelling.
