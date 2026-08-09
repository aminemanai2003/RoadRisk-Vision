# Benchmark protocol

Release benchmark hardware is an NVIDIA RTX 2050 4 GB, Intel i5-11400H and
16 GB RAM. Use a licensed 1080p/30 H.264 fixture with pinned models/configuration.

Record source duration, normalized duration, wall-clock duration, peak VRAM,
peak process RAM, dropped/decoded frames, model revisions and package commit.
The v0.1 gate is ≤5× source duration, ≤3.5 GB VRAM and ≤8 GB process RAM.

CUDA and video codecs are not byte-deterministic. Acceptance means stable schema,
IDs and ordering plus tolerance-bounded numeric results—not identical MP4 bytes.
