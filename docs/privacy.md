# Privacy

Default analysis is local after an explicit model download.

- Input audio, metadata and absolute source paths are omitted.
- Annotated output is silent.
- Absolute GPS is excluded unless the user explicitly enables location export.
- Run manifests contain source basename, SHA-256, versions and configuration.
- Raw videos, GPS traces, model weights and `runs/` are ignored by Git.

Derived events can still reveal behavior and timing. Treat complete run folders
as personal telematics data, obtain driver/passenger consent, and review local
recording laws before collecting or sharing footage.
