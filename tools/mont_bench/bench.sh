#!/usr/bin/env bash
# usage: bench.sh <worktree> <csv> <nvcc extra flags or ""> <tag> <args...> [ -- <tag> <args...> ]...
# Uploads the worktree (tracked + new files), builds once with the given flags, runs each config.
set -euo pipefail
WT="$1"; CSV="$2"; EXTRA="$3"; shift 3
export PATH=$HOME/.venv/bin:$PATH
TMP=$(mktemp -d); trap 'rm -rf $TMP' EXIT
HASH="$(git -C "$WT" rev-parse --short=10 HEAD)"; git -C "$WT" diff --quiet HEAD -- || HASH+="-dirty"
( cd "$WT" && { git ls-files -z; git ls-files -z --others --exclude-standard; } | tar --null -T - -czf $TMP/cgbn.tar.gz )
attempt() {  # the colab CLI occasionally fails transiently (upload or ssh); retry instead of dying silently
  local n
  for n in 1 2 3; do
    if colab upload -s cgbn-bench $TMP/cgbn.tar.gz content/cgbn_b.tar.gz >$TMP/up.log 2>&1 && remote "$@" >$TMP/out.log 2>$TMP/err.log && grep -q "^### " $TMP/out.log; then
      cat $TMP/out.log; return 0
    fi
    echo "BENCH-ERROR: attempt $n failed: $(tail -n 2 $TMP/up.log 2>/dev/null | tr '\n' ' ') $(tail -n 3 $TMP/err.log 2>/dev/null | tr '\n' ' ')" >&2
    sleep 15
  done
  return 1
}
remote() {
colab ssh -s cgbn-bench -- bash -s -- "$@" <<'REMOTE'
set -euo pipefail
export PATH="/usr/local/cuda/bin:$PATH" GIT_HASH="$1"
EXTRA="$2"; shift 2
export NVCCFLAGS="-O3 -std=c++14 -Xcompiler -pthread $EXTRA"
rm -rf /content/cgbn_b /content/cgbn_b.csv && mkdir -p /content/cgbn_b && tar -xzf /content/cgbn_b.tar.gz -C /content/cgbn_b
D=/content/cgbn_b/samples/sample_05_mont_sqr
BUILD=""
while (($#)); do
  TAG="$1"; shift; ARGS=()
  while (($#)) && [[ "$1" != "--" ]]; do ARGS+=("$1"); shift; done
  (($#)) && shift
  echo "### $TAG : ${ARGS[*]}"
  bash $D/run_benchmark.sh --csv /content/cgbn_b.csv --tag "$TAG" $BUILD -- "${ARGS[@]}" 2>&1 | grep -a -E "rror|Avg kernel|match|MISMATCH|FAIL|mismatch" | cut -c1-300 || echo "RUN FAILED"
  BUILD="--no-build"
done
REMOTE
}
attempt "$HASH" "$EXTRA" "$@" || { echo "BENCH-ERROR: giving up" >&2; exit 1; }
colab download -s cgbn-bench content/cgbn_b.csv $TMP/r.csv >/dev/null
if [[ -s "$CSV" ]]; then tail -n +2 $TMP/r.csv >> "$CSV"; else cp $TMP/r.csv "$CSV"; fi
