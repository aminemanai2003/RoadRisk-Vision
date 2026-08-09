# Product specification

RoadRisk Vision analyzes a completed phone recording on a PC. The phone is not
tethered to the PC and no Android inference application is planned for V1.

## V1 success criteria

1. A fresh Python 3.11 environment can install the package and run the mock
   end-to-end test without network access.
2. H.264/H.265 phone footage is normalized to a silent, rotation-corrected,
   constant-frame-rate video.
3. Real inference combines HybridNets D3 road/lane perception with YOLOX-S
   multi-class detections and sequential GPU execution.
4. Uncalibrated runs never invent metric distance or TTC.
5. Every completed run contains validated, versioned artifacts.
6. RTX 2050 release benchmark stays under 3.5 GB VRAM, 8 GB RAM and five times
   the source duration for 1080p/30 input.

## Non-goals

- Live Android analysis or an in-car PC.
- Vehicle actuation, steering advice or autonomous-driving claims.
- Premium, claim-frequency or claim-severity modelling.
- Uploading footage or telemetry to a hosted service.
