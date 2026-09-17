"""CLI wrapper for the privacy scan (kairos.privacy). Used by CI and the deploy scripts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kairos.privacy import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
