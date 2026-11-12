#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || (cd "$SCRIPT_DIR/../.." && pwd))"

CSV_FILE="${CSV_FILE:-${SCRIPT_DIR}/benchmark_results.csv}"
ARCH="${ARCH:-}"
TAG="${TAG:-}"
BENCH_BIN="${BENCH_BIN:-${SCRIPT_DIR}/mont_sqr}"
DO_BUILD=1
BENCH_ARGS=()

print_usage() {
  cat <<EOF
Usage: $(basename "$0") [wrapper options] [--] [mont_sqr options]

Compiles samples/sample_05_mont_sqr/mont_sqr, prints the current git hash,
runs the benchmark + CPU verification, and appends the result(s) to a CSV file.

Wrapper options:
  --arch <target|sm_XX>  Makefile target (kepler, maxwell, pascal, volta, turing,
                         ampere, ada, hopper) or CUDA arch (e.g. sm_61, sm_86).
                         Defaults to auto-detecting via nvidia-smi (or \$ARCH).
  --tag <label>          Optional tag to label runs sharing a git hash (or \$TAG).
  -o, --csv <path>       CSV file to append results to (or \$CSV_FILE).
                         Default: ${SCRIPT_DIR}/benchmark_results.csv
  --no-build             Skip compilation and run existing binary.
  -h, --help             Show this help message.

Benchmark options (forwarded to mont_sqr):
  -n <instances>         Number of random 1018-bit instances (default: 102400)
  -i <iterations>        Montgomery squarings per instance (default: 256)
  -r <runs>              Number of timed GPU kernel runs (default: 5)
  -t <tpi>               Threads per instance: 8, 16, or 32 (default: 8)
  -a                     Benchmark and verify all supported TPIs (8, 16, 32)
  -s <seed>              Random seed (default: 1)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)
      [[ $# -ge 2 ]] || { echo "Error: --arch requires an argument" >&2; exit 1; }
      ARCH="$2"
      shift 2
      ;;
    --arch=*)
      ARCH="${1#*=}"
      shift
      ;;
    --tag)
      [[ $# -ge 2 ]] || { echo "Error: --tag requires a label" >&2; exit 1; }
      TAG="$2"
      shift 2
      ;;
    --tag=*)
      TAG="${1#*=}"
      shift
      ;;
    -o|--csv)
      [[ $# -ge 2 ]] || { echo "Error: $1 requires a file path" >&2; exit 1; }
      CSV_FILE="$2"
      shift 2
      ;;
    --csv=*)
      CSV_FILE="${1#*=}"
      shift
      ;;
    --no-build)
      DO_BUILD=0
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    --)
      shift
      BENCH_ARGS+=("$@")
      break
      ;;
    *)
      BENCH_ARGS+=("$1")
      shift
      ;;
  esac
done

detect_gpu_arch() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi

  local cc
  cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]' || true)"
  if [[ ! "$cc" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
    return 1
  fi

  local major="${BASH_REMATCH[1]}"
  local minor="${BASH_REMATCH[2]}"
  local sm_exact="sm_${major}${minor}"

  if command -v nvcc >/dev/null 2>&1 && nvcc --help 2>/dev/null | grep -q "'${sm_exact}'"; then
    echo "$sm_exact"
    return 0
  fi

  case "${major}.${minor}" in
    3.*)       echo "kepler" ;;
    5.*)       echo "maxwell" ;;
    6.*)       echo "pascal" ;;
    7.0|7.2)   echo "volta" ;;
    7.*)       echo "turing" ;;
    8.9)       echo "ada" ;;
    8.*)       echo "ampere" ;;
    9.*|10.*|12.*) echo "hopper" ;;
    *)         echo "$sm_exact" ;;
  esac
}

if [[ -z "${GIT_HASH:-}" ]]; then
  if git -C "$REPO_ROOT" rev-parse --verify HEAD >/dev/null 2>&1; then
    GIT_HASH="$(git -C "$REPO_ROOT" rev-parse --short=10 HEAD)"
    if ! git -C "$REPO_ROOT" diff --quiet HEAD --; then
      GIT_HASH="${GIT_HASH}-dirty"
    fi
  else
    GIT_HASH="unknown"
  fi
fi

if [[ -n "$TAG" ]]; then
  echo "Git hash: ${GIT_HASH} (tag: ${TAG})"
else
  echo "Git hash: ${GIT_HASH}"
fi

if [[ "$DO_BUILD" -eq 1 ]]; then
  if [[ -z "$ARCH" ]]; then
    if ! ARCH="$(detect_gpu_arch)"; then
      echo "Error: could not auto-detect GPU architecture via nvidia-smi." >&2
      echo "Specify --arch <kepler|maxwell|pascal|volta|turing|ampere|ada|hopper|sm_XX> or set ARCH=..." >&2
      exit 1
    fi
  fi

  if [[ "$ARCH" == sm_* || "$ARCH" == "native" ]]; then
    echo "Compiling mont_sqr (SM=${ARCH}) ..."
    make -C "$SCRIPT_DIR" build SM="$ARCH"
  else
    echo "Compiling mont_sqr (target=${ARCH}) ..."
    make -C "$SCRIPT_DIR" "$ARCH"
  fi
else
  ARCH="${ARCH:-prebuilt}"
  if [[ ! -x "$BENCH_BIN" ]]; then
    echo "Error: benchmark binary not found or not executable: ${BENCH_BIN}" >&2
    exit 1
  fi
fi

TMP_OUT="$(mktemp)"
trap 'rm -f "$TMP_OUT"' EXIT

TIMESTAMP="$(date -Iseconds)"

echo "Running ${BENCH_BIN} ${BENCH_ARGS[*]+"${BENCH_ARGS[*]}"} ..."
"$BENCH_BIN" ${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"} | tee "$TMP_OUT"

CSV_ROWS="$(BENCH_TAG="$TAG" awk \
  -v ts="$TIMESTAMP" \
  -v hash="$GIT_HASH" \
  -v arch="$ARCH" \
  '
  BEGIN {
    tag = ENVIRON["BENCH_TAG"]
    gsub(/\r/, "", tag)
    gsub(/"/, "\"\"", tag)
    if (tag ~ /[,"\n]/) {
      tag = "\"" tag "\""
    }
  }
  /^GPU[[:space:]]*:/ {
    gpu = $0
    sub(/^GPU[[:space:]]*:[[:space:]]*/, "", gpu)
    gsub(/\r$/, "", gpu)
    gsub(/"/, "\"\"", gpu)
    if (gpu ~ /[,"]/) {
      gpu = "\"" gpu "\""
    }
  }
  /^Instances[[:space:]]*:/ {
    instances = $3
    bits = $4
    gsub(/[^0-9]/, "", bits)
  }
  /^Launch config[[:space:]]*:/ {
    tpi = $4
    gsub(/[^0-9]/, "", tpi)
  }
  /^Squarings\/inst[[:space:]]*:/ {
    iterations = $3
  }
  /^Benchmark runs[[:space:]]*:/ {
    runs = $4
  }
  /^[[:space:]]*Avg kernel time[[:space:]]*:/ {
    avg_ms = $5
    best_ms = $8
  }
  /^[[:space:]]*Avg throughput[[:space:]]*:/ {
    avg_tp = $4
    best_tp = $8
  }
  /^[[:space:]]*Avg time\/sqr[[:space:]]*:/ {
    ns_per_sqr = $4
  }
  /^All [0-9]+ results match!/ {
    cpu_s = $11
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,PASS\n", \
      ts, hash, tag, gpu, arch, bits, tpi, instances, iterations, runs, \
      avg_ms, best_ms, avg_tp, best_tp, ns_per_sqr, cpu_s
    rows++
  }
  END {
    if (rows == 0) {
      exit 1
    }
  }
  ' "$TMP_OUT")" || {
  echo "Error: failed to parse verified benchmark metrics from mont_sqr output." >&2
  exit 1
}

CSV_HEADER="timestamp,git_hash,tag,gpu,arch,bits,tpi,instances,iterations,runs,avg_ms,best_ms,avg_msqr_per_s,best_msqr_per_s,ns_per_sqr,cpu_s,verified"
mkdir -p "$(dirname "$CSV_FILE")"
if [[ ! -s "$CSV_FILE" ]]; then
  echo "$CSV_HEADER" > "$CSV_FILE"
elif [[ "$(head -n 1 "$CSV_FILE")" != "$CSV_HEADER" ]]; then
  echo "Warning: CSV header in ${CSV_FILE} does not match expected schema (${CSV_HEADER})" >&2
fi
printf "%s\n" "$CSV_ROWS" >> "$CSV_FILE"

ROW_COUNT="$(printf "%s\n" "$CSV_ROWS" | wc -l | tr -d '[:space:]')"
echo "Appended ${ROW_COUNT} benchmark row(s) for git ${GIT_HASH}${TAG:+ (tag: ${TAG})} to ${CSV_FILE}"
