#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_ROOT/pbs_common.sh"
cd "$REPO_ROOT"

echo "Running Python script..."

# python3 -m SPARBackdoor.dataset_generation.load_beavertails
python3 -m SPARBackdoor.dataset_generation.dataset_craft
