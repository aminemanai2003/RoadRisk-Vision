# Synthetic integration fixtures

Fixtures are generated—not downloaded—by `scripts/generate_synthetic_fixtures.py`.
The generated test patterns and black frames are dedicated to the public domain
under CC0-1.0. They contain no people, faces, plates, audio, GPS, identifiers or
captured-world metadata.

Generated files cover H.264 MP4, 90-degree rotation metadata, variable frame
rate, a deliberately corrupt stream, and a solid black zero-object scenario.
They are created inside pytest temporary directories and never committed.
