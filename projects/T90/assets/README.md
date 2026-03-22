# T90 Assets

This directory stores deployment-side auxiliary artifacts.

- `t90_casebase.csv`
  Default historical guidance database used by the online recommender at the factory site.
- `t90_casebase.parquet`
  Optional high-efficiency version for environments that already provide `pyarrow` or `fastparquet`.

The recommended workflow is:

1. Build or refresh the casebase from private raw data with `test.py`.
2. Deliver the CSV file together with the interface package as the default runtime asset.
3. Use the parquet file only when the target environment has a verified parquet engine.
4. Let `interface.py` load the CSV file by default when no explicit `casebase_path` is provided.
