#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

LOCAL_CSV="${CSV_FILE:-${REPO_ROOT}/benchmark_results.csv}"
SESSION="${SESSION:-}"
GPU="${GPU:-T4}"
ARCH="${ARCH:-}"
TAG="${TAG:-}"
STOP_AFTER=0
BENCH_ARGS=()

print_usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [--] [mont_sqr options]

Tars git-tracked files in ${REPO_ROOT}, reuses an active Colab GPU session
(or creates a new one), uploads and untars the archive, compiles and runs the
mont_sqr benchmark on the Colab GPU, downloads the resulting CSV, and appends
it to a local CSV file.

Options:
  -s, --session <name>   Colab session name to reuse or create. If omitted,
                         reuses the first active GPU session or creates
                         'cgbn-bench'.
  --gpu <type>           GPU accelerator when creating a new session:
                         T4, L4, G4, A100, H100 (default: ${GPU}).
  --arch <target|sm_XX>  CUDA arch / Makefile target override on Colab
                         (default: auto-detected via nvidia-smi).
  --tag <label>          Optional tag to label runs sharing a git hash (or \$TAG).
  -o, --csv <path>       Local CSV file to append results to
                         (default: ${LOCAL_CSV}).
  --stop                 Stop the Colab session on exit if created by this run.
  -h, --help             Show this help message.

Benchmark options (forwarded to mont_sqr):
  -n <instances>         Number of random 1018-bit instances (default: 102400)
  -i <iterations>        Montgomery squarings per instance (default: 256)
  -r <runs>              Number of timed GPU kernel runs (default: 5)
  -t <tpi>               Threads per instance: 8, 16, or 32 (default: 8)
  -a                     Benchmark and verify all supported TPIs (8, 16, 32)
  -s <seed>              Pass after '--' (e.g. -- -s 42) so it isn't parsed as --session
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--session)
      [[ $# -ge 2 ]] || { echo "Error: $1 requires a session name" >&2; exit 1; }
      SESSION="$2"
      shift 2
      ;;
    --session=*)
      SESSION="${1#*=}"
      shift
      ;;
    --gpu)
      [[ $# -ge 2 ]] || { echo "Error: --gpu requires a GPU type (T4, L4, G4, A100, H100)" >&2; exit 1; }
      GPU="$2"
      shift 2
      ;;
    --gpu=*)
      GPU="${1#*=}"
      shift
      ;;
    --arch)
      [[ $# -ge 2 ]] || { echo "Error: --arch requires a target or sm_XX" >&2; exit 1; }
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
      [[ $# -ge 2 ]] || { echo "Error: $1 requires a CSV file path" >&2; exit 1; }
      LOCAL_CSV="$2"
      shift 2
      ;;
    --csv=*)
      LOCAL_CSV="${1#*=}"
      shift
      ;;
    --stop)
      STOP_AFTER=1
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

# Resolve `colab` CLI (supports PATH install, ~/.venv/bin/colab, or `uv run colab`)
if [[ -n "${COLAB_BIN:-}" ]]; then
  COLAB=("$COLAB_BIN")
elif command -v colab >/dev/null 2>&1; then
  COLAB=(colab)
elif [[ -x "${HOME}/.venv/bin/colab" ]]; then
  COLAB=("${HOME}/.venv/bin/colab")
elif command -v uv >/dev/null 2>&1 && uv run colab version >/dev/null 2>&1; then
  COLAB=(uv run colab)
else
  echo "Error: 'colab' CLI not found. Install with: uv tool install google-colab-cli" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
CREATED_SESSION=0

cleanup() {
  rm -rf "$TMP_DIR"
  if [[ "$STOP_AFTER" -eq 1 && "$CREATED_SESSION" -eq 1 && -n "$SESSION" ]]; then
    echo "Stopping Colab session '${SESSION}' ..."
    "${COLAB[@]}" stop -s "$SESSION" || true
  fi
}
trap cleanup EXIT

# 1. Compute local git hash (carried over to Colab since .git is not tarred)
GIT_HASH="$(git -C "$REPO_ROOT" rev-parse --short=10 HEAD)"
if ! git -C "$REPO_ROOT" diff --quiet HEAD --; then
  GIT_HASH="${GIT_HASH}-dirty"
fi
if [[ -n "$TAG" ]]; then
  echo "Local git hash: ${GIT_HASH} (tag: ${TAG})"
else
  echo "Local git hash: ${GIT_HASH}"
fi

# 2. Tar all git-tracked files in the repository
LOCAL_TAR="$TMP_DIR/cgbn.tar.gz"
FILE_LIST="$TMP_DIR/files.nul"
: > "$FILE_LIST"

while IFS= read -r -d '' rel; do
  [[ -e "$REPO_ROOT/$rel" ]] && printf '%s\0' "$rel" >> "$FILE_LIST"
done < <(git -C "$REPO_ROOT" ls-files -z)

tar -C "$REPO_ROOT" --null -T "$FILE_LIST" -czf "$LOCAL_TAR"
echo "Created source archive ($(wc -c < "$LOCAL_TAR" | tr -d '[:space:]') bytes)"

# 3. Find an active Colab GPU session or create a new one
SESSIONS_OUT="$("${COLAB[@]}" sessions)"

if [[ -n "$SESSION" ]]; then
  if printf '%s\n' "$SESSIONS_OUT" | grep -Fq "[${SESSION}] "; then
    echo "Using existing Colab session '${SESSION}'"
  else
    echo "Creating Colab session '${SESSION}' (--gpu ${GPU}) ..."
    "${COLAB[@]}" new -s "$SESSION" --gpu "$GPU"
    CREATED_SESSION=1
  fi
else
  # Look for an active locally-named GPU session: `[name] ... | Hardware: X | ... | Variant: GPU`
  FOUND_LINE="$(printf '%s\n' "$SESSIONS_OUT" | awk '
    /^\[[^?][^]]*\].*\| Variant: GPU(\r|$| )/ { print; exit }
  ')"
  if [[ -n "$FOUND_LINE" ]]; then
    SESSION="${FOUND_LINE#\[}"
    SESSION="${SESSION%%\]*}"
    echo "Found active Colab GPU session '${SESSION}': ${FOUND_LINE}"
  else
    SESSION="cgbn-bench"
    echo "No active Colab GPU session found; creating '${SESSION}' (--gpu ${GPU}) ..."
    "${COLAB[@]}" new -s "$SESSION" --gpu "$GPU"
    CREATED_SESSION=1
  fi
fi

# 4. Upload archive to Colab
REMOTE_TAR="content/cgbn.tar.gz"
REMOTE_CSV="content/cgbn_benchmark.csv"
echo "Uploading archive to Colab session '${SESSION}' ..."
"${COLAB[@]}" upload -s "$SESSION" "$LOCAL_TAR" "$REMOTE_TAR"

# 5. Untar, compile, and run the benchmark binary on Colab
REMOTE_RUNNER="${REPO_ROOT}/samples/sample_05_mont_sqr/colab_remote_run.sh"
echo "Untarring, compiling, and running benchmark on Colab session '${SESSION}' ..."
"${COLAB[@]}" ssh -s "$SESSION" -- \
  bash -s -- "$GIT_HASH" "$TAG" "$ARCH" "/${REMOTE_CSV}" ${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"} \
  < "$REMOTE_RUNNER"

# 6. Download the resulting CSV and append to local CSV
DOWNLOADED_CSV="$TMP_DIR/downloaded_results.csv"
"${COLAB[@]}" download -s "$SESSION" "$REMOTE_CSV" "$DOWNLOADED_CSV"

if [[ ! -s "$DOWNLOADED_CSV" ]]; then
  echo "Error: downloaded CSV is empty." >&2
  exit 1
fi

mkdir -p "$(dirname "$LOCAL_CSV")"
if [[ ! -s "$LOCAL_CSV" ]]; then
  cp "$DOWNLOADED_CSV" "$LOCAL_CSV"
  ADDED_ROWS="$(($(wc -l < "$DOWNLOADED_CSV") - 1))"
else
  EXISTING_HEADER="$(head -n 1 "$LOCAL_CSV")"
  DOWNLOADED_HEADER="$(head -n 1 "$DOWNLOADED_CSV")"
  if [[ "$EXISTING_HEADER" != "$DOWNLOADED_HEADER" ]]; then
    echo "Warning: CSV header in ${LOCAL_CSV} does not match downloaded header (${DOWNLOADED_HEADER})" >&2
  fi
  tail -n +2 "$DOWNLOADED_CSV" >> "$LOCAL_CSV"
  ADDED_ROWS="$(tail -n +2 "$DOWNLOADED_CSV" | wc -l | tr -d '[:space:]')"
fi

echo "Appended ${ADDED_ROWS} row(s) from Colab session '${SESSION}'${TAG:+ (tag: ${TAG})} to ${LOCAL_CSV}"
