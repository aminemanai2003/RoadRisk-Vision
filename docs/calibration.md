# Camera calibration

Metric distance and TTC are optional. Without a compatible profile the fields
remain null and the UI shows qualitative warnings only.

Mount the phone rigidly, keep the same camera/lens, orientation, stabilization,
zoom and crop, then identify a measured rectangle on the road plane. Pass four
points in this order: near-left, near-right, far-right, far-left.

```powershell
roadrisk calibrate frame-source.mp4 --frame-ms 1500 `
  --camera-id phone-main --height-m 1.25 `
  --reference-length-m 20 --reference-width-m 3.5 `
  --corners "240,980;1680,980;1250,520;670,520" `
  --output calibrations/phone-main.json
```

Road distance uses the detected object's bottom-center contact point. Estimates
are suppressed when the point is outside the road mask, geometry is unstable,
the crop/aspect ratio is incompatible, or confidence is low. TTC additionally
requires a stable closing trend; stationary or receding objects use
`not_closing`.

Profiles with residual above 5% of the measured reference length are rejected.
Four-point homography has an exact reference fit; future checkerboard validation
may populate a non-zero residual and lens intrinsics.
