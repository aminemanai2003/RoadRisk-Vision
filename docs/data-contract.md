# Data contract — schema version 1

Readers must reject a newer `schema_version` major. New optional fields may be
added within version 1. Unknown fields are rejected by the reference Pydantic
models so compatibility changes stay deliberate.

## Shared conventions

- `frame_index`: zero-based integer in normalized-video order.
- `video_time_ms`: integer milliseconds from normalized-video start.
- boxes: floating-point `x1,y1,x2,y2` pixels in normalized output coordinates.
- masks: same raster dimensions as the normalized frame while in memory.
- stored units: metres, seconds and metres/second. km/h is display-only.
- `run_id`: UTC timestamp plus source SHA-256 prefix.
- `track_id`: positive integer, stable only inside one run.
- `event_id`: UUID5 of run, event type, start time and sorted track IDs.

Nullable numeric values use one of: `not_calibrated`,
`incompatible_calibration`, `unstable_estimate`, `missing_telemetry`,
`low_confidence`, or `not_closing`.

## events.jsonl

One finalized event per line, ordered by `(start_time_ms,event_id)`:

```json
{
  "schema_version": 1,
  "event_id": "uuid",
  "run_id": "20260809T120000000000Z-deadbeef",
  "event_type": "rapid_closing",
  "warning_state": "slow_down",
  "severity": 2,
  "start_time_ms": 1200,
  "end_time_ms": 2600,
  "track_ids": [7],
  "min_ttc_s": 3.4,
  "min_distance_m": 14.2,
  "evidence": {"rule_version": 1}
}
```

Event types are `object_in_path`, `rapid_closing`, `collision_risk` and
`lane_departure`. Warning states are `normal`, `monitor`, `slow_down`,
`brake_risk`, `lane_departure` and `hazard`.

Lane-departure evidence includes `direction`, segmentation confidence,
ego-boundary clearance ratio, normalized lateral offset and fitted boundary
positions. It contains no claim of driver intent and is emitted only after the
same temporal event-duration rule as other risk events.

## Trip summary

JSON stores duration, valid GPS distance, GPS coverage, event counts, events per
hour and optionally events per 100 km. The CSV is one flattened row with event
counts named `event_count_<type>`. A per-100-km rate is emitted only with at
least 80% telemetry coverage and 1 km of valid distance.

These are `risk_events`, not insurance claims and not automatically near misses.
