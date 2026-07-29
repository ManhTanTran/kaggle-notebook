"""Run the offline smoke test."""

import sys

from chunkbench.cli import main

if __name__ == "__main__":
    sys.argv.extend(
        ["--config", "configs/experiments/all_qa_datasets_smoke.yaml"]
    )
    main()
