# T90 Assets

This directory stores the delivery-side auxiliary assets for the stage-aware T90 recommender.

- `t90_casebase.csv`
  The default 50-minute DCS casebase used by all stages.
- `t90_casebase_ph120.csv`
  The PH(120min)-augmented casebase used only when the current stage policy enables PH.
- `t90_stage_policy.json`
  The precomputed DCS-only stage identifier, stage policy, and PH enable/disable rules used at runtime.
- `t90_casebase.parquet`
  Optional parquet copy of the base casebase for environments that already provide `pyarrow` or `fastparquet`.
- `t90_casebase_ph120.parquet`
  Optional parquet copy of the PH-augmented casebase for the same environments.

Recommended workflow:

1. Refresh these assets from private offline data with [build_v2_delivery_assets.py](/D:/PSE/博兴京博/BXJingBo/projects/T90/dev/build_v2_delivery_assets.py).
2. Deliver the CSV and JSON assets together with `core/`, `interface.py`, `example.py`, and `README.md`.
3. Keep CSV as the default runtime path. Use parquet only when the target environment has a verified parquet engine.
