#!/usr/bin/env bash
set -euo pipefail

accelerate_args=()
while (( $# )) && [[ "$1" != -- ]]; do
    accelerate_args+=("$1")
    shift
done
if (( $# < 2 )); then
    echo "Usage: bash modules/pretrain/launch.sh [accelerate options] -- config.yaml [overrides...]" >&2
    exit 2
fi
shift

module=modules.pretrain.src.run_pretrain
python -m "$module" "$@" dataset.prepare_only=true
exec accelerate launch "${accelerate_args[@]}" --module "$module" "$@" dataset.prepare_only=false
