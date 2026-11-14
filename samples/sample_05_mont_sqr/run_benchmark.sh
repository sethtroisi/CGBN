#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || (cd "$SCRIPT_DIR/../.." && pwd))"

CSV_FILE="${CSV_FILE:-$SCRIPT_DIR/benchmark_results.csv}"
ARCH="${ARCH:-}"
TAG="${TAG:-}"
BENCH_BIN="${BENCH_BIN:-$SCRIPT_DIR/mont_sqr}"
DO_BUILD=1
LEGACY_CSV=0
BENCH_ARGS=()

die() { echo "Error: $*" >&2; exit 1; }

usage() {
  cat <<EOF
Usage: $(basename "$0") [wrapper options] [--] [mont_sqr options]

Builds mont_sqr, runs GPU benchmark + GMP verification (mont_sqr and/or mont_mul),
and appends CSV row(s).

Wrapper options:
  --arch <target|sm_XX>  Makefile target (kepler..blackwell) or sm_XX (default: auto-detect)
  --tag <label>          Optional label for runs sharing a git hash (or \$TAG)
  -o, --csv <path>       Output CSV path (default: $CSV_FILE)
  --legacy-csv           Emit 17-column legacy mont_sqr CSV schema
  --no-build             Skip compilation and run existing binary
  -h, --help             Show this help message

Forwarded mont_sqr options:
  -m <sqr|mul|both>  -b <512|1024|2048>  -u <input_bits>  -f
  -n <instances>     -i <iterations>     -r <runs>        -t <4|8|16|32>  -a  -s <seed>
EOF
}

while (($#)); do
  case "$1" in
    --arch=*|--tag=*|--csv=*) set -- "${1%%=*}" "${1#*=}" "${@:2}" ;;
    --arch)       [[ $# -ge 2 ]] || die "$1 requires an argument"; ARCH="$2"; shift 2 ;;
    --tag)        [[ $# -ge 2 ]] || die "$1 requires a label"; TAG="$2"; shift 2 ;;
    -o|--csv)     [[ $# -ge 2 ]] || die "$1 requires a file path"; CSV_FILE="$2"; shift 2 ;;
    --legacy-csv) LEGACY_CSV=1; shift ;;
    --no-build)   DO_BUILD=0; shift ;;
    -h|--help)    usage; exit 0 ;;
    --)           shift; BENCH_ARGS+=("$@"); break ;;
    *)            BENCH_ARGS+=("$1"); shift ;;
  esac
done

detect_gpu_arch() {
  local cc sm
  cc="$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader 2>/dev/null)" || return 1
  cc="${cc//[[:space:]]/}"
  [[ "$cc" =~ ^([0-9]+)\.([0-9]+)$ ]] || return 1
  sm="sm_${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
  case "$cc" in
    3.*)     echo kepler ;;
    5.*)     echo maxwell ;;
    6.*)     echo pascal ;;
    7.0|7.2) echo volta ;;
    7.*)     echo turing ;;
    8.9)     echo ada ;;
    8.*)     echo ampere ;;
    9.0)     echo hopper ;;
    12.0)    if nvcc --list-gpu-code 2>/dev/null | grep -qx sm_120; then echo blackwell; else echo hopper; fi ;;
    *)       if nvcc --list-gpu-code 2>/dev/null | grep -qx "$sm"; then echo "$sm"; else echo hopper; fi ;;
  esac
}

if [[ -z "${GIT_HASH:-}" ]]; then
  if GIT_HASH="$(git -C "$REPO_ROOT" rev-parse --short=10 HEAD 2>/dev/null)"; then
    git -C "$REPO_ROOT" diff --quiet HEAD -- || GIT_HASH+="-dirty"
  else
    GIT_HASH="unknown"
  fi
fi
echo "Git hash: ${GIT_HASH}${TAG:+ (tag: $TAG)}"

if ((DO_BUILD)); then
  [[ -n "$ARCH" ]] || ARCH="$(detect_gpu_arch)" ||
    die "Could not auto-detect GPU arch via nvidia-smi; specify --arch <target|sm_XX>"
  echo "Compiling mont_sqr ($ARCH) ..."
  if [[ "$ARCH" == sm_* || "$ARCH" == native ]]; then
    make -C "$SCRIPT_DIR" build SM="$ARCH"
  else
    make -C "$SCRIPT_DIR" "$ARCH"
  fi
else
  ARCH="${ARCH:-prebuilt}"
  [[ -x "$BENCH_BIN" ]] || die "Benchmark binary not found or not executable: $BENCH_BIN"
fi

LEGACY_HEADER="timestamp,git_hash,tag,gpu,arch,bits,tpi,instances,iterations,runs,avg_ms,best_ms,avg_msqr_per_s,best_msqr_per_s,ns_per_sqr,cpu_s,verified"
NEW_HEADER="timestamp,git_hash,tag,op,gpu,arch,env_bits,bits,tpi,instances,iterations,runs,avg_ms,best_ms,avg_mops_per_s,best_mops_per_s,ns_per_op,cpu_s,verified"

if [[ -s "$CSV_FILE" && "$(head -n 1 "$CSV_FILE")" == "$LEGACY_HEADER" ]]; then
  LEGACY_CSV=1
fi

TMP_OUT="$(mktemp)"
trap 'rm -f "$TMP_OUT"' EXIT

echo "Running $BENCH_BIN ${BENCH_ARGS[*]+"${BENCH_ARGS[*]}"} ..."
"$BENCH_BIN" ${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"} | tee "$TMP_OUT"

CSV_ROWS="$(BENCH_TAG="$TAG" awk -v ts="$(date -Iseconds)" -v hash="$GIT_HASH" -v arch="$ARCH" -v legacy="$LEGACY_CSV" '
  function esc(s) { gsub(/\r/, "", s); gsub(/"/, "\"\"", s); return (s ~ /[,"\n]/) ? "\"" s "\"" : s }
  BEGIN { raw_tag = ENVIRON["BENCH_TAG"]; op = "mont_sqr" }
  /^GPU[ \t]*:/             { sub(/^[^:]*:[ \t]*/, ""); gpu = esc($0) }
  /^Operation[ \t]*:/       { sub(/^[^:]*:[ \t]*/, ""); gsub(/[ \t\r]/, "", $0); sub(/^cgbn_/, "", $0); op = $0 }
  /^Instances[ \t]*:/       { instances = $3; bits = $4; env_bits = $8; gsub(/[^0-9]/, "", bits); gsub(/[^0-9]/, "", env_bits) }
  /^Launch config[ \t]*:/   { tpi = $4; gsub(/[^0-9]/, "", tpi) }
  /^(Squarings|Multiplies)\/inst[ \t]*:/ { iterations = $3 }
  /^Benchmark runs[ \t]*:/  { runs = $4 }
  /Avg kernel time[ \t]*:/  { avg_ms = $5; best_ms = $8 }
  /Avg throughput[ \t]*:/   { avg_tp = $4; best_tp = $8 }
  /Avg time\/(sqr|mul)[ \t]*:/ { ns_per_op = $4 }
  /^All [0-9]+ results match!/ {
    if (legacy == 1) {
      eff_tag = raw_tag
      if (op != "mont_sqr" && eff_tag !~ /mul/) {
        eff_tag = (eff_tag == "") ? op : (op ":" eff_tag)
      }
      printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,PASS\n",
        ts, hash, esc(eff_tag), gpu, arch, bits, tpi, instances, iterations, runs,
        avg_ms, best_ms, avg_tp, best_tp, ns_per_op, $11
    } else {
      printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,PASS\n",
        ts, hash, esc(raw_tag), op, gpu, arch, env_bits, bits, tpi, instances, iterations, runs,
        avg_ms, best_ms, avg_tp, best_tp, ns_per_op, $11
    }
    rows++
  }
  END { exit (rows == 0) }
' "$TMP_OUT")" || die "Failed to parse verified benchmark metrics from mont_sqr output"

CSV_HEADER="$NEW_HEADER"
if ((LEGACY_CSV)); then
  CSV_HEADER="$LEGACY_HEADER"
fi

mkdir -p "$(dirname "$CSV_FILE")"
if [[ ! -s "$CSV_FILE" ]]; then
  echo "$CSV_HEADER" > "$CSV_FILE"
elif [[ "$(head -n 1 "$CSV_FILE")" != "$CSV_HEADER" ]]; then
  echo "Warning: CSV header in $CSV_FILE does not match expected schema" >&2
fi
printf '%s\n' "$CSV_ROWS" >> "$CSV_FILE"
echo "Appended $(grep -c . <<<"$CSV_ROWS") row(s) for git ${GIT_HASH}${TAG:+ (tag: $TAG)} to $CSV_FILE"
