# Safety and limitations

RoadRisk Vision performs retrospective research analysis. Its warning symbols
are annotations, not live driver instructions. It must not be connected to
braking, steering or acceleration systems.

Known limitations include night/rain performance, camera vibration, occlusion,
unusual roads, model/domain bias, inaccurate user calibration and monocular
depth uncertainty. Detection absence never proves a road is safe.

Do not market results as certified ADAS, collision predictions or autonomous
driving. Validate new rules on licensed, representative datasets and document
false-positive and false-negative behavior.

## Lane-departure event definition

The system samples both detected lane boundaries over the lower 45% of the
frame, fits their positions, and treats the image center as the fixed ego-camera
reference. Evidence is accepted only when both boundaries occur on at least 35%
of sampled rows and their apparent width is plausible. A departure signal occurs
when the ego reference is inside the closest 12% of a boundary or beyond it and
persists for the configured minimum event duration (350 ms by default).

This is an experimental retrospective event. Curves, merges, missing paint,
camera misalignment and non-centered mounting can create errors. The reported
left/right value describes observed boundary proximity; it is not steering
advice.
