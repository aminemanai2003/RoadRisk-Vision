# Model card

## System

HybridNets D3 provides BDD100K-derived road/lane perception. YOLOX-S provides
COCO object detections for person, bicycle, car, motorcycle, bus and truck.
ByteTrack associates detections over time; deterministic rules create events.

## Intended use

Offline experimentation, annotated trip review, computer-vision education and
privacy-preserving telematics feature research.

## Out-of-scope use

Live safety intervention, autonomous control, law enforcement, identity
recognition, fault determination, insurance pricing, or decisions about a person
without independent validation and appropriate governance.

## Evaluation

Upstream benchmark claims do not establish RoadRisk Vision performance on phone
footage. Project releases must publish pinned-hardware latency/memory results and
scenario-level errors. Until that report exists, real inference is experimental.
