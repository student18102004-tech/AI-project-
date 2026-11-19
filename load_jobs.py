
from pathlib import Path

from database.db import (
    init_db,
    init_jobs_table,
    import_jobs_from_csv,
    count_jobs
)

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "datasets" / "jobs_dataset.csv"

init_db()
init_jobs_table()

import_jobs_from_csv(CSV_PATH)

print("Jobs dataset imported successfully!")
print("Total jobs:", count_jobs())
