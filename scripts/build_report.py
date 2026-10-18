"""Build the experiment report PDF from the saved results (no API calls).

    python scripts/build_report.py      # writes docs/experiment_report.pdf
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from douroub.pdf_report import build  # noqa: E402

if __name__ == "__main__":
    print(f"Saved {build()}")
