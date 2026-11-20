"""Download selected public benchmarks from the exact manuscript revisions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

from .itemgen import TRACK


def verified_download(entry):
    target = TRACK / entry["file"]
    expected_hash, expected_bytes = entry["sha256"], entry["bytes"]
    if target.exists():
        with target.open("rb") as cache:
            if target.stat().st_size == expected_bytes and hashlib.file_digest(cache, "sha256").hexdigest() == expected_hash:
                return
        raise ValueError(f"Local file fails frozen checksum: {entry['file']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    request = Request(entry["url"], headers={"User-Agent": "T1-visual-item-audit"})
    digest, n = hashlib.sha256(), 0
    with NamedTemporaryFile(dir=target.parent, delete=False) as pending:
        tmp = Path(pending.name)
        try:
            with urlopen(request, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    n += len(chunk)
                    if n > expected_bytes:
                        raise ValueError("Download exceeded the frozen file size")
                    digest.update(chunk)
                    pending.write(chunk)
            if n != expected_bytes or digest.hexdigest() != expected_hash:
                raise ValueError(f"Downloaded bytes differ: {entry['file']}")
            pending.close()
            tmp.replace(target)
        finally:
            tmp.unlink(missing_ok=True)


def main():
    sources = json.loads((TRACK / "frozen" / "external_sources.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suites", nargs="+", choices=sources)
    for suite in parser.parse_args().suites:
        for entry in sources[suite]["downloads"]:
            verified_download(entry)
            print(f"Verified {entry['file']}", flush=True)


if __name__ == "__main__":
    main()
