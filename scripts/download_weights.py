"""Install the fixed MobileSAM checkpoint; verify before replacing any cache file."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.pin_dinov2 import sha256_file

MOBILE_SAM_URL = "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/b01a9ccef3b9e10b099b544efe004d0871802c3b/weights/mobile_sam.pt"
MOBILE_SAM_SHA256 = "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f"
MOBILE_SAM_MAX_BYTES = 40_728_226


def checkpoint_valid(path: Path) -> bool:
    return path.is_file() and sha256_file(path) == MOBILE_SAM_SHA256


def install_checkpoint(target: Path) -> Path:
    target = target.expanduser()
    if checkpoint_valid(target):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".mobile-sam-", delete=False) as output:
            temporary = Path(output.name)
            with urlopen(MOBILE_SAM_URL, timeout=60) as response:
                total = 0
                while chunk := response.read(min(1024 * 1024, MOBILE_SAM_MAX_BYTES + 1 - total)):
                    total += len(chunk)
                    if total > MOBILE_SAM_MAX_BYTES:
                        raise ValueError("MobileSAM response exceeds the expected checkpoint size")
                    output.write(chunk)
        if not checkpoint_valid(temporary):
            raise ValueError("MobileSAM checksum mismatch; the existing checkpoint was preserved")
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(
        os.environ.get("MOBILE_SAM_CHECKPOINT") or Path.home() / ".cache/mobile_sam/mobile_sam.pt"
    ))
    args = parser.parse_args()
    try:
        print(f"MobileSAM verified: {install_checkpoint(args.output)}")
    except (OSError, ValueError) as exc:
        print(f"MobileSAM installation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
