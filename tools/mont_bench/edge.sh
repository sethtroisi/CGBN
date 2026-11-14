#!/usr/bin/env bash
# usage: edge.sh <worktree> [edge args...]   (uploads tracked+new files, builds and runs the edge test on the colab session)
set -euo pipefail
WT="$1"; shift
export PATH=$HOME/.venv/bin:$PATH
TMP=$(mktemp -d); trap 'rm -rf $TMP' EXIT
( cd "$WT" && { git ls-files -z; git ls-files -z --others --exclude-standard; } | tar --null -T - -czf $TMP/cgbn.tar.gz )
colab upload -s cgbn-bench $TMP/cgbn.tar.gz content/cgbn_edge.tar.gz
remote() {
colab ssh -s cgbn-bench -- bash -s -- "${EDGE_EXTRA:-}" "$@" <<'REMOTE'
set -euo pipefail
export PATH="/usr/local/cuda/bin:$PATH"
EDGE_EXTRA="$1"; shift
rm -rf /content/cgbn_edge && mkdir -p /content/cgbn_edge && tar -xzf /content/cgbn_edge.tar.gz -C /content/cgbn_edge
cd /content/cgbn_edge/samples/sample_06_mont_sqr_edge
time make build EXTRA="$EDGE_EXTRA" 2>&1 | grep -v "^W:" | grep -a -E "error|Error|warning: .*ptxas|register" | cut -c1-300 | head -30 || true
./mont_sqr_edge "$@"
REMOTE
}
# the colab CLI sometimes reports "Already-active SSH session (HTTP 429)" for a while after a previous run
for attempt in 1 2 3 4 5; do
  if remote "$@" > $TMP/out.log 2>&1 && ! grep -q "HTTP 429" $TMP/out.log; then cat $TMP/out.log; exit 0; fi
  if ! grep -q "HTTP 429" $TMP/out.log; then cat $TMP/out.log; exit 1; fi
  echo "EDGE-RETRY: colab ssh busy (attempt $attempt)" >&2; sleep 20
done
echo "EDGE-ERROR: giving up" >&2; exit 1
