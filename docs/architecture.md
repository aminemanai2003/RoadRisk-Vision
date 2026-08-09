# Architecture

RoadRisk Vision is a local, bounded-memory pipeline. `pipeline/analyzer.py` owns
the run lifecycle and connects independent adapters:

1. `io/video.py` validates MP4/MOV H.264/H.265, removes audio and metadata,
   applies rotation and normalizes VFR to CFR.
2. `perception/` runs HybridNets D3 and YOLOX-S sequentially. Backends are lazy:
   importing the package never downloads or loads weights.
3. `tracking/` assigns per-run IDs. Real inference uses official BYTETracker;
   the deterministic IoU tracker is restricted to mocks and tests.
4. `geometry/` projects an object's bottom-center road-contact point through an
   optional road-plane calibration and estimates temporal closing speed.
5. `risk/` converts evidence into advisory, deterministic events.
6. `rendering/` writes a silent annotated H.264 video through FFmpeg.
7. `io/artifacts.py` validates and atomically publishes the completed run.

GPU stages are intentionally sequential. An out-of-memory condition fails the
run with `GPU_OOM`; the process never silently switches devices mid-run.

Run directories start as `<run_id>.partial`. Success atomically renames them to
`<run_id>`; failure and cancellation retain only useful diagnostics and remove
invalid video intermediates. V1 does not resume partial runs.
