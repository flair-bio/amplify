#!/usr/bin/env bash
# scripts/setup_binaries.sh

# Exit immediately if a command exits with a non-zero status.
set -euo pipefail

echo "🚀 Setting up external bioinformatics binaries..."

SEQKIT_VERSION="2.9.0"
PV_VERSION="1.12.0"

# Honor UV_PROJECT_ENVIRONMENT so the binaries follow the venv location uv uses
# (e.g. a node-local path on shared-filesystem clusters). Defaults to ./.venv.
VENV_DIR="${UV_PROJECT_ENVIRONMENT:-${VIRTUAL_ENV:-.venv}}"

# Ensure the virtual environment exists first!
if [ ! -d "$VENV_DIR/bin" ]; then
  echo "Error: Virtual environment not found at '$VENV_DIR'. Please run 'uv sync' first." >&2
  echo "If you set UV_PROJECT_ENVIRONMENT, ensure it points to your project's venv." >&2
  exit 1
fi

# Resolve to an absolute path so it stays valid after we cd into TMP_DIR.
VENV_BIN="$(cd "$VENV_DIR/bin" && pwd)"

# Platform & Architecture Detection
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$ARCH" in
  x86_64|amd64)  ARCH_NORM="amd64" ;;
  arm64|aarch64) ARCH_NORM="arm64" ;;
  *) echo "Error: Unsupported architecture: $ARCH" >&2; exit 1 ;;
esac

case "$OS" in
  Darwin)
    SEQKIT_OS="darwin"
    MMSEQS_BUILD="osx-universal"
    ;;
  Linux)
    SEQKIT_OS="linux"
    case "$ARCH_NORM" in
      amd64)
        if grep -q avx2 /proc/cpuinfo 2>/dev/null; then
          MMSEQS_BUILD="linux-avx2"
        elif grep -q sse4_1 /proc/cpuinfo 2>/dev/null; then
          MMSEQS_BUILD="linux-sse41"
        else
          MMSEQS_BUILD="linux-sse2"
        fi
        ;;
      arm64) MMSEQS_BUILD="linux-arm64" ;;
    esac
    ;;
  *) echo "Error: Unsupported operating system: $OS" >&2; exit 1 ;;
esac

MMSEQS_URL="https://mmseqs.com/latest/mmseqs-${MMSEQS_BUILD}.tar.gz"
SEQKIT_URL="https://github.com/shenwei356/seqkit/releases/download/v${SEQKIT_VERSION}/seqkit_${SEQKIT_OS}_${ARCH_NORM}.tar.gz"

if command -v curl >/dev/null 2>&1; then
  DOWNLOAD=(curl -fsSL)
elif command -v wget >/dev/null 2>&1; then
  DOWNLOAD=(wget -qO-)
else
  echo "Error: Install curl or wget to download the binaries." >&2
  exit 1
fi

for required_command in tar make cc; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    echo "Error: Required command '$required_command' was not found." >&2
    echo "macOS: run 'xcode-select --install'. Linux: install build tools and zlib development headers." >&2
    exit 1
  fi
done

download_extract() {
  "${DOWNLOAD[@]}" "$1" | tar xzf -
}

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
cd "$TMP_DIR"

# ---------------------------------------------------------
# 1. Install MMseqs2
# ---------------------------------------------------------
echo "📦 Downloading MMseqs2 ($MMSEQS_BUILD)..."
download_extract "$MMSEQS_URL"

# ---------------------------------------------------------
# 2. Install SeqKit
# ---------------------------------------------------------
echo "📦 Downloading SeqKit (v${SEQKIT_VERSION})..."
download_extract "$SEQKIT_URL"

# ---------------------------------------------------------
# 3. Install pigz (Requires gcc/make & zlib-dev)
# ---------------------------------------------------------
echo "📦 Compiling pigz from source..."
download_extract "https://zlib.net/pigz/pigz.tar.gz"
(cd pigz*/ && make -s)

# ---------------------------------------------------------
# 4. Install pv
# ---------------------------------------------------------
echo "📦 Downloading and building pv (v${PV_VERSION})..."
download_extract "https://www.ivarch.com/programs/sources/pv-${PV_VERSION}.tar.gz"
(cd "pv-$PV_VERSION" && ./configure -q && make -s)

echo "Installing binaries to $VENV_BIN..."
mv mmseqs/bin/mmseqs seqkit pigz*/pigz "pv-$PV_VERSION/pv" "$VENV_BIN/"
ln -sf pigz "$VENV_BIN/unpigz"

echo "✅ All binaries installed successfully to $VENV_BIN!"
echo "You can now use them directly via 'uv run', e.g., 'uv run mmseqs -h'"
