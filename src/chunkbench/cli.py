"""Command-line entry point."""

import argparse
import json
from pathlib import Path

from chunkbench.common.logging import configure_logging
from chunkbench.config.loader import load_config
from chunkbench.pipeline.matrix import ExperimentMatrixRunner
from chunkbench.pipeline.runner import BenchmarkRunner


def main() -> None:
    """Run a configured benchmark and print a compact summary."""
    parser = argparse.ArgumentParser(description="Run chunkbench experiment")
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    args = parser.parse_args()
    configure_logging()
    config = load_config(Path(args.config))
    if config.matrix:
        result = ExperimentMatrixRunner(config, Path.cwd()).run()
    else:
        result = BenchmarkRunner(config, Path.cwd()).run()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
