

import sys
from pathlib import Path

# Make src/ importable without requiring `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent / "src"))

from casecomp_judge.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
