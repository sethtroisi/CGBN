#!/usr/bin/env bash
# usage: unit.sh <worktree> [gtest args]   -- builds and runs CGBN's unit_tests/tester on the colab session
set -euo pipefail
WT="$1"; shift
export PATH=$HOME/.venv/bin:$PATH
TMP=$(mktemp -d); trap 'rm -rf $TMP' EXIT
( cd "$WT" && git ls-files -z | tar --null -T - -czf $TMP/cgbn.tar.gz )
remote() {
colab ssh -s cgbn-bench -- bash -s -- "$@" <<'REMOTE'
set -euo pipefail
export PATH="/usr/local/cuda/bin:$PATH"
if [ ! -d /usr/src/googletest/googletest ]; then apt-get update -qq >/dev/null 2>&1 || true; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libgtest-dev >/dev/null; fi
rm -rf /content/cgbn_u && mkdir -p /content/cgbn_u && tar -xzf /content/cgbn_u.tar.gz -C /content/cgbn_u
cd /content/cgbn_u/unit_tests
export GTEST_HOME=/usr/src/googletest/googletest
( time make ampere ) 2>&1 | grep -a -E "error|real|Error" | grep -v "instantiation of\|cgbn_error" | cut -c1-250 | head -20
./tester "$@" 2>&1 | grep -a -E "^\[  (FAILED|PASSED)|^\[==========\]|FAILED|OK \]" | grep -a -E "FAILED|PASSED|==========|mont|bn2mont" | cut -c1-160 | tail -150
REMOTE
}
for attempt in 1 2 3 4 5; do
  if colab upload -s cgbn-bench $TMP/cgbn.tar.gz content/cgbn_u.tar.gz > $TMP/up.log 2>&1 && remote "$@" > $TMP/out.log 2>&1 && ! grep -q "HTTP 429" $TMP/out.log; then cat $TMP/out.log; exit 0; fi
  if ! grep -q "HTTP 429" $TMP/out.log $TMP/up.log 2>/dev/null; then cat $TMP/up.log $TMP/out.log; exit 1; fi
  echo "UNIT-RETRY ($attempt)" >&2; sleep 20
done
exit 1
