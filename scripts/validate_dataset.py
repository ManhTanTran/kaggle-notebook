"""Validate a configured dataset adapter."""

import argparse

from chunkbench.data.validation import validate_dataset
from chunkbench.registry.datasets import build_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="synthetic")
args = parser.parse_args()
bundle = build_dataset(args.dataset).load()
validate_dataset(bundle)
print(f"Valid: {args.dataset} ({len(bundle.documents)} documents)")

