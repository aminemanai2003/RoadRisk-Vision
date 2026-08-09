# Models and provenance

Analysis never downloads weights. `models.lock.json` pins official release URLs,
sizes, SHA-256 checksums, source revisions and inference sizes.

```powershell
roadrisk models download
roadrisk models verify
python -m pip install -e ".[inference]"
python -m pip install "git+https://github.com/Megvii-BaseDetection/YOLOX.git@e1052df71842031413f6030723c3607b839c80ce"
```

- HybridNets D3: road and lane segmentation at 640×384.
- YOLOX-S COCO: selected road-user classes at 640×640, confidence 0.35 and NMS
  IoU 0.45.

YOLOX adds multiclass road users missing from the upstream one-class HybridNets
BDD100K configuration. The cost is latency, so the models run sequentially and
the release gate is 3.5 GB peak VRAM and five times source duration on RTX 2050.

Model weights are excluded from Git and retain their upstream licenses/terms.
