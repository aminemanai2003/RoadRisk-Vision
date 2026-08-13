# Camera calibration

Two independent profiles exist. The **road-plane profile** maps image points to
ground distance and is what the analysis pipeline consumes. The **lens profile**
records the camera's intrinsics and distortion, and is optional.

Metric distance and TTC are optional. Without a compatible profile the fields
remain null and the UI shows qualitative warnings only.

## Road plane (four points)

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
Four-point homography has an exact reference fit, so its residual is zero by
construction; the checkerboard workflow below is what produces a measured one.

The local dashboard includes an interactive **Calibrate camera** workspace. It
decodes the chosen frame locally, labels the four click targets in order,
validates point order/convexity/area, previews the polygon, and writes the same
versioned profile used by the CLI. No frame is uploaded or saved by the UI.

## Lens distortion (checkerboard)

Optional, and separate from the road-plane profile above — neither requires the
other. It estimates the camera matrix and distortion coefficients so frames can
be undistorted before any geometric reasoning.

```powershell
roadrisk calibrate-lens calibration-shots/ `
  --camera-id phone-main --lens-id wide `
  --board-columns 9 --board-rows 6 --square-size-m 0.025 `
  --output calibrations/phone-main-lens.json
```

`--board-columns`/`--board-rows` count **inner** corners, so a printed board of
10x7 squares is 9x6. `--square-size-m` is the measured side of one printed
square; measure it after printing, since printer scaling is rarely exact.

### Capture technique

- Print the board and mount it on something rigid and flat. Paper that bows on a
  wall or held in hand is the most common source of a bad fit.
- Shoot **stills**, not video frames — rolling shutter and compression smear the
  corners. Use the same camera, lens, zoom, crop and orientation you will record
  with; a different resolution produces a profile the frames are rejected against.
- 15-20 views. Fewer than 5 detected boards is refused outright.
- Vary the pose between shots: tilt roughly 20-40 degrees around both axes, and
  move the board through the centre, all four corners and the edges of the frame.
  Distortion is strongest at the edges, so a set of centred head-on shots leaves
  it badly constrained even when the residual looks good.
- Keep the board sharp and evenly lit. Avoid glare, motion blur and shots where
  the board is cut off — a partially visible board is silently skipped.

### Residual and limitations

The residual is the RMS reprojection error in pixels. Profiles above **1.0 px**
are rejected; override with `--max-residual-px` only if you understand why yours
is higher. A low residual means the model fits *the images you supplied* — it is
not evidence that the pose set covered the frame, so it can look healthy and
still extrapolate poorly at the edges.

The estimate covers the standard five-coefficient radial/tangential model only.
Fisheye and heavily cropped or digitally stabilized footage are out of scope.
The profile is tied to one resolution and orientation; a rotated or resized
frame is rejected rather than rescaled.

The lens profile is not yet wired into `analyze` — it is written, validated and
usable via `LensProfile.undistort`, with pipeline integration to follow.

No road footage or external dataset is needed to exercise this: the tests
generate checkerboard images from a known synthetic camera and run on CPU.
