#!/usr/bin/env bash
# r2_build.sh — build transcribe.cpp (CPU-only) into transcribe-cli, reproducibly.
#
# Usage:
#   git clone https://github.com/handy-computer/transcribe.cpp.git transcribe.cpp
#   bash scripts/r2_build.sh              # idempotent: skips configure/build when nothing changed
#   bash scripts/r2_build.sh --clean      # wipe the scratch copy and rebuild from scratch
#
# Conventions:
#   * the source tree next to this repo is treated as read-only; all work happens in a scratch copy;
#   * the build directory never lives inside the git work tree;
#   * no model download, no GPU/CUDA.
#
# Override any of these with environment variables:
#   SRC_SRC    path to the transcribe.cpp checkout   (default: ./transcribe.cpp)
#   BUILD_SRC  scratch copy + build root             (default: $TMPDIR/nemotron-build/tc-src)
#   JOBS       parallel build jobs                   (default: nproc)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SRC_SRC="${SRC_SRC:-$ROOT/transcribe.cpp}"                    # read-only reference checkout
SCRATCH="${SCRATCH:-${TMPDIR:-/tmp}/nemotron-build}"
BUILD_SRC="${BUILD_SRC:-$SCRATCH/tc-src}"                     # writable copy
BUILD_DIR="${BUILD_DIR:-$BUILD_SRC/build}"
JOBS="${JOBS:-$(nproc)}"
PATCH_DIR="$ROOT/patches/r2"                                  # every *.patch applied in filename order
EVIDENCE="${EVIDENCE:-$ROOT/build/build.json}"

CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

[ -d "$SRC_SRC" ] || {
  echo "FAIL: source not found at $SRC_SRC" >&2
  echo "      git clone https://github.com/handy-computer/transcribe.cpp.git \"$SRC_SRC\"" >&2
  exit 1
}

# CPU-only. Switch names follow CMakeLists.txt (TRANSCRIBE_* maps to GGML_*);
# TRANSCRIBE_CUDA already defaults to OFF, pinned here so a stray env var cannot flip it.
CMAKE_ARGS=(
  -S "$BUILD_SRC"
  -B "$BUILD_DIR"
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DTRANSCRIBE_CUDA=OFF
  -DTRANSCRIBE_METAL=OFF
  -DTRANSCRIBE_VULKAN=OFF
  -DTRANSCRIBE_HIP=OFF
  -DTRANSCRIBE_BUILD_TESTS=OFF
  -DTRANSCRIBE_BUILD_EXAMPLES=ON
  -DTRANSCRIBE_BUILD_TOOLS=OFF
  -DTRANSCRIBE_X86_CONSERVATIVE=OFF
)

# A stamp over (cmake args, jobs, patch bytes) — changes to any of them force a rebuild.
stamp() {
  {
    printf '%s\n' "${CMAKE_ARGS[@]}"
    printf 'jobs=%s\n' "$JOBS"
    if [ -d "$PATCH_DIR" ]; then
      find "$PATCH_DIR" -maxdepth 1 -name '*.patch' -print0 | sort -z | xargs -0 -r cat
    fi
  } | sha256sum | cut -d' ' -f1
}

STAMP="$(stamp)"
STAMP_FILE="$BUILD_DIR/.r2_stamp"

echo "== r2_build =="
echo "  source : $SRC_SRC (-> $BUILD_SRC)"
echo "  build  : $BUILD_DIR"

if [ "$CLEAN" = 1 ]; then
  echo "  --clean: removing $BUILD_SRC"
  rm -rf "$BUILD_SRC"
fi

# 1) Copy the source (idempotent: reuse the existing scratch copy to avoid a ~100 MB copy each run)
if [ ! -d "$BUILD_SRC" ]; then
  cp -a "$SRC_SRC" "$BUILD_SRC"
  echo "  copied source tree"
fi

# 2) Apply patches.
#    Must be idempotent: the scratch copy persists, so a re-run would otherwise hit the patch
#    again. `patch` in a non-interactive shell silently skips "Reversed (or previously applied)"
#    and looks like success — so detect "already applied" with a reverse dry-run first, and hard
#    fail on a genuine conflict instead of compiling a half-patched tree.
if [ -d "$PATCH_DIR" ]; then
  for p in "$PATCH_DIR"/*.patch; do
    [ -e "$p" ] || continue
    if patch -p1 -d "$BUILD_SRC" -R --dry-run -f -i "$p" >/dev/null 2>&1; then
      echo "  already applied: $(basename "$p")"
      continue
    fi
    echo "  apply patch: $(basename "$p")"
    if ! patch -p1 -d "$BUILD_SRC" -f --no-backup-if-mismatch -i "$p"; then
      echo "  ERROR: patch did not apply (upstream source may have moved): $p" >&2
      echo "         try 'scripts/r2_build.sh --clean' to re-copy from $SRC_SRC" >&2
      exit 1
    fi
  done
fi

# 3) Configure + build
if [ -x "$BUILD_DIR/bin/transcribe-cli" ] && [ -f "$STAMP_FILE" ] && [ "$(cat "$STAMP_FILE")" = "$STAMP" ]; then
  echo "  up-to-date (stamp match), skipping configure/build"
  ELAPSED="0"
else
  cmake "${CMAKE_ARGS[@]}" >/dev/null
  echo "  building with -j$JOBS ..."
  START=$(date +%s.%N)
  cmake --build "$BUILD_DIR" -j"$JOBS"
  END=$(date +%s.%N)
  ELAPSED=$(awk -v s="$START" -v e="$END" 'BEGIN{printf "%.1f", e-s}')
  echo "  elapsed: ${ELAPSED}s"
  printf '%s' "$STAMP" > "$STAMP_FILE"
fi

BIN="$BUILD_DIR/bin/transcribe-cli"
[ -x "$BIN" ] || { echo "FAIL: $BIN not found" >&2; exit 1; }

# 4) Self-check the binary
echo "== verify =="
HELP_HEAD="$("$BIN" --help 2>&1 | head -n 3 | tr '\n' '|')"
echo "  --help  : ${HELP_HEAD}"
"$BIN" --version >/dev/null 2>&1 \
  && echo "  --version: $("$BIN" --version 2>&1 | head -n 1)" \
  || echo "  --version: not implemented (prints usage instead — expected)"
MISSING="$("$BIN" -m /tmp/__r2_no_such_model.gguf /tmp/__r2_no_such_audio.wav 2>&1 | tail -n 1 || true)"
echo "  missing-model: $MISSING"
case "$MISSING" in
  *"file not found"*) ;;
  *) echo "FAIL: a missing model did not produce the expected error" >&2; exit 1 ;;
esac

# 5) Write build evidence
TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
mkdir -p "$(dirname "$EVIDENCE")"
cat > "$EVIDENCE" <<JSON
{
  "generated_at_utc": "$TIMESTAMP",
  "source_dir": "$SRC_SRC",
  "build_dir": "$BUILD_DIR",
  "cmake_args": $(printf '%s\n' "${CMAKE_ARGS[@]}" | python3 -c 'import json,sys;print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))'),
  "jobs": "$JOBS",
  "binaries": {"transcribe-cli": "$BIN"},
  "elapsed_s": $(printf '%s' "$ELAPSED" | grep -E '^[0-9.]+$' || echo 0),
  "cpu_only": true,
  "gpu_backend": "none",
  "cli_help_head": $(printf '%s' "$HELP_HEAD" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')
}
JSON
python3 -c "import json;json.load(open('$EVIDENCE'))"
echo "  evidence: $EVIDENCE"
echo "== done =="
