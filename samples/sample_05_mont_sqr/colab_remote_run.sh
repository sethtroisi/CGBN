#!/usr/bin/env bash
# Runs inside a Google Colab VM (typically piped via `colab ssh -s <sess> -- bash -s -- ...`).
# Usage: colab_remote_run.sh <git_hash> <tag> <arch> <remote_csv> [mont_sqr options...]
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <git_hash> <tag> <arch> <remote_csv> [mont_sqr options...]" >&2
  exit 1
fi

export GIT_HASH="$1"
export TAG="$2"
export ARCH="$3"
REMOTE_CSV="$4"
shift 4

export PATH="/usr/local/cuda/bin:$PATH"

rm -rf /content/cgbn "$REMOTE_CSV"
mkdir -p /content/cgbn
tar -xzf /content/cgbn.tar.gz -C /content/cgbn

if ! printf '#include <gmp.h>\n' | g++ -E -x c++ - >/dev/null 2>&1; then
  echo "Installing libgmp-dev on Colab VM ..."
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libgmp-dev >/dev/null
fi

bash /content/cgbn/samples/sample_05_mont_sqr/run_benchmark.sh --csv "$REMOTE_CSV" "$@"
