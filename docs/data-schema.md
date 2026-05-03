# Data Schema

- Bronze: immutable raw packets, events, health, video, metadata.
- Silver: decoded/validated Parquet tables.
- Gold: ML-ready windowed datasets with manifest, stats, normalization, train/val/test splits.

Canonical message schema: `libs/rfpose-schemas/proto/csi.proto`.
