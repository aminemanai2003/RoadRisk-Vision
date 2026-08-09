"""Local RoadRisk Vision review dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from roadrisk_vision.dashboard import (
    CORNER_LABELS,
    annotate_calibration_frame,
    read_video_frame,
    scale_display_click,
    validate_corner_order,
)
from roadrisk_vision.geometry import CalibrationProfile
from roadrisk_vision.pipeline import AnalysisOptions, analyze_video

st.set_page_config(page_title="RoadRisk Vision", page_icon="🚘", layout="wide")

st.title("RoadRisk Vision")
st.caption("Private, post-drive AI dashcam analysis on this PC")
st.warning(
    "Research and driver-awareness software only. It does not control a vehicle or replace "
    "an attentive driver."
)


def existing_path(value: str) -> Path | None:
    value = value.strip()
    return Path(value) if value else None


def render_calibration_workspace() -> None:
    st.header("Interactive camera calibration")
    st.write(
        "Everything stays local. Select: near-left, near-right, far-right, far-left. "
        "Use a measured rectangle painted or marked on the road plane."
    )
    left, right = st.columns([2, 1])
    with right:
        video_value = st.text_input("Calibration video path")
        frame_ms = st.number_input("Frame time (ms)", min_value=0, value=1500, step=100)
        camera_id = st.text_input("Camera ID", value="phone-main")
        lens_id = st.text_input("Lens ID", value="main-1x")
        mount_height = st.number_input("Mount height (m)", min_value=0.1, value=1.25)
        reference_width = st.number_input("Measured width (m)", min_value=0.1, value=3.5)
        reference_length = st.number_input("Measured length (m)", min_value=0.1, value=20.0)
        output_value = st.text_input("Profile output path", value="calibration.json")
        reset = st.button("Reset points")

    selection_key = f"{video_value}|{int(frame_ms)}"
    if reset or st.session_state.get("calibration_source") != selection_key:
        st.session_state.calibration_source = selection_key
        st.session_state.calibration_points = []
    points: list[tuple[float, float]] = st.session_state.get("calibration_points", [])
    video = existing_path(video_value)
    if video is None or not video.is_file():
        left.info("Select an existing local video to load the calibration frame.")
        return
    try:
        frame = read_video_frame(video, int(frame_ms))
    except ValueError as exc:
        left.error(str(exc))
        return
    height, width = frame.shape[:2]
    display_width = min(1000, width)
    annotated = annotate_calibration_frame(frame, points)
    with left:
        st.caption(
            f"Next point: {CORNER_LABELS[len(points)] if len(points) < 4 else 'complete'}"
        )
        click = streamlit_image_coordinates(
            annotated,
            width=display_width,
            key=f"calibration-click-{selection_key}-{len(points)}",
        )
    if click and len(points) < 4:
        point = scale_display_click(
            click["x"],
            click["y"],
            original_width=width,
            display_width=display_width,
        )
        st.session_state.calibration_points = [*points, point]
        st.rerun()
    valid, message = validate_corner_order(points, (width, height))
    if valid:
        right.success(message)
    else:
        right.warning(message)
    right.dataframe(
        [
            {"order": index + 1, "point": CORNER_LABELS[index], "x": x, "y": y}
            for index, (x, y) in enumerate(points)
        ],
        hide_index=True,
        use_container_width=True,
    )
    if right.button("Save calibration profile", type="primary", disabled=not valid):
        profile = CalibrationProfile.create(
            camera_id=camera_id,
            lens_id=lens_id,
            width=width,
            height=height,
            mount_height_m=mount_height,
            reference_width_m=reference_width,
            reference_length_m=reference_length,
            corners=points,
            calibration_frame_ms=int(frame_ms),
        )
        profile.save(Path(output_value))
        right.success(f"Saved locally to {output_value}")


workspace = st.sidebar.radio("Workspace", ["Analyze", "Calibrate camera"])
if workspace == "Calibrate camera":
    render_calibration_workspace()
    st.stop()


with st.sidebar:
    st.header("Analysis setup")
    video_value = st.text_input("Phone video path", placeholder=r"C:\Trips\drive.mp4")
    output_value = st.text_input("Run output folder", value="runs")
    telemetry_value = st.text_input("Optional GPX/CSV telemetry")
    calibration_value = st.text_input("Optional calibration profile")
    backend = st.selectbox("Perception backend", ["hybrid", "mock"])
    device = st.selectbox("Compute device", ["auto", "cuda", "cpu"])
    include_location = st.checkbox("Include absolute location in exports", value=False)
    analyze_clicked = st.button("Analyze recording", type="primary", use_container_width=True)

if "run_path" not in st.session_state:
    st.session_state.run_path = None

if analyze_clicked:
    video = existing_path(video_value)
    if video is None or not video.is_file():
        st.error("Select an existing phone video.")
    else:
        with st.status("Analyzing recording", expanded=True) as status:
            st.write("Normalizing rotation, frame rate, metadata and audio…")
            try:
                result = analyze_video(
                    AnalysisOptions(
                        video=video,
                        output=Path(output_value),
                        telemetry=existing_path(telemetry_value),
                        calibration=existing_path(calibration_value),
                        device=device,
                        backend=backend,
                        include_location=include_location,
                    )
                )
            except Exception as exc:
                status.update(label="Analysis failed", state="error")
                st.exception(exc)
            else:
                st.session_state.run_path = result
                status.update(label="Analysis complete", state="complete")

run_path = st.session_state.run_path
if run_path:
    run_path = Path(run_path)
    manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_path / "trip_summary.json").read_text(encoding="utf-8"))
    events_path = run_path / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]

    first, second, third, fourth = st.columns(4)
    first.metric("Duration", f"{summary['duration_s'] / 60:.1f} min")
    second.metric("Risk events", sum(summary["event_counts"].values()))
    third.metric("Events/hour", f"{summary['risk_events_per_hour']:.2f}")
    fourth.metric(
        "Valid distance",
        f"{summary['valid_distance_km']:.1f} km" if summary["valid_distance_km"] else "No GPS",
    )

    video_column, provenance_column = st.columns([2, 1])
    with video_column:
        st.subheader("Annotated recording")
        st.video(str(run_path / "annotated.mp4"))
    with provenance_column:
        st.subheader("Run provenance")
        st.json(manifest, expanded=False)

    st.subheader("Risk timeline")
    timeline = pd.read_json(run_path / "timeline.json")
    if not timeline.empty:
        chart = timeline.set_index("video_time_ms")[["severity", "object_count"]]
        st.line_chart(chart)

    st.subheader("Events")
    st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)

    st.subheader("Exports")
    export_columns = st.columns(4)
    for column, name in zip(
        export_columns,
        ["events.jsonl", "trip_summary.json", "trip_summary.csv", "manifest.json"],
        strict=True,
    ):
        path = run_path / name
        column.download_button(
            name,
            path.read_bytes(),
            file_name=name,
            mime="application/json" if path.suffix != ".csv" else "text/csv",
            use_container_width=True,
        )
else:
    st.info("Choose a local phone recording to start, or use the mock backend for a UI demo.")
    st.subheader("Processing states")
    st.markdown(
        "Input selected → calibration readiness → queued → running → completed. "
        "Cancelled and failed runs retain a privacy-safe diagnostic manifest."
    )
