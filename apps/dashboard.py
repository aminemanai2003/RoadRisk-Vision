"""Local RoadRisk Vision review dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

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
