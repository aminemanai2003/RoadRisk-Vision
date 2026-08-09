# Benchmark protocol

Release benchmark hardware is an NVIDIA RTX 2050 4 GB, Intel i5-11400H and
16 GB RAM. Use a licensed 1080p/30 H.264 fixture with pinned models/configuration.

Record source duration, normalized duration, wall-clock duration, peak VRAM,
peak process RAM, dropped/decoded frames, model revisions and package commit.
The v0.1 gate is ≤5× source duration, ≤3.5 GB VRAM and ≤8 GB process RAM.

CUDA and video codecs are not byte-deterministic. Acceptance means stable schema,
IDs and ordering plus tolerance-bounded numeric results—not identical MP4 bytes.

Generate the CC0 fixture and run the measured pipeline with:

```powershell
python scripts/generate_benchmark_video.py .cache/benchmark-1080p30.mp4
roadrisk benchmark .cache/benchmark-1080p30.mp4 `
  --fixture-license CC0-1.0 --device cuda --backend hybrid `
  --output docs/benchmark-results/rtx2050
```

The command samples process-plus-child RSS, reads PyTorch peak allocated VRAM,
records the fixture/model/commit provenance, and exits with code 2 when any gate
fails. Mock or CPU runs are useful for testing the harness but cannot pass the
CUDA/dual-model release gate.
