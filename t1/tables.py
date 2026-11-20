"""Regenerate paper tables from the frozen aggregate inputs."""
from __future__ import annotations

import subprocess
import sys

from .itemgen import TRACK

BUILDERS = (
    "build_tables",
    "build_holdout_followup",
    "build_english_followup",
    "build_full_external_tables",
    "build_muir_confirmation",
    "build_followup_tables",
    "build_pixel_transform_table",
    "build_review_updates",
    "build_template_five",
)


def main():
    for builder in BUILDERS:
        path = TRACK / "paper_ieeeaccess" / f"{builder}.py"
        subprocess.run([sys.executable, str(path)], check=True, cwd=TRACK)


if __name__ == "__main__":
    main()
