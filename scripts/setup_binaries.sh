#!/bin/bash
# scripts/setup_binaries.sh

# Exit immediately if a command exits with a non-zero status.
set -e

echo "🚀 Setting up external bioinformatics binaries..."

# Define paths
# Honor UV_PROJECT_ENVIRONMENT so the binaries follow the venv location uv uses
# (e.g. a node-local path on shared-filesystem clusters). Defaults to ./.venv.
VENV_DIR="${UV_PROJECT_ENVIRONMENT:-${VIRTUAL_ENV:-.venv}}"
TMP_DIR="$(mktemp -d)"

# Ensure the virtual environment exists first!
if [ ! -d "$VENV_DIR/bin" ]; then
  echo "Error: Virtual environment not found at '$VENV_DIR'. Please run 'uv sync' first."
  echo "If you set UV_PROJECT_ENVIRONMENT, ensure it points to your project's venv."
  exit 1
fi

# Resolve to an absolute path so it stays valid after we cd into TMP_DIR.
VENV_BIN="$(cd "$VENV_DIR/bin" && pwd)"

# Create a temporary workspace
mkdir -p $TMP_DIR
cd $TMP_DIR

# ---------------------------------------------------------
# 1. Install MMseqs2
# ---------------------------------------------------------
echo "📦 Downloading MMseqs2..."
wget -qO- https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz | tar xz
mv mmseqs/bin/mmseqs "$VENV_BIN/"

# ---------------------------------------------------------
# 2. Install SeqKit
# ---------------------------------------------------------
echo "📦 Downloading SeqKit..."
wget -qO- https://github.com/shenwei356/seqkit/releases/download/v2.9.0/seqkit_linux_amd64.tar.gz | tar xz
mv seqkit "$VENV_BIN/"

# ---------------------------------------------------------
# 3. Install pigz (Requires gcc/make on the system)
# ---------------------------------------------------------
echo "📦 Compiling pigz from source..."
wget -qO- https://zlib.net/pigz/pigz.tar.gz | tar xz

# FIX: Properly separate the cd, make, and mv commands
cd pigz*/
make -s
mv pigz unpigz "$VENV_BIN/"
cd ..

# ---------------------------------------------------------
# 4. Install pv (Requires gcc/make on the system)
# ---------------------------------------------------------
echo "📦 Compiling pv from source..."
wget -qO- https://www.ivarch.com/programs/sources/pv-1.8.5.tar.gz | tar xz

cd pv-*/
# pv requires configuration before it can be compiled
./configure -q
make -s
# Move the compiled binary directly to the venv, avoiding the need for 'make install'
mv pv "$VENV_BIN/"
cd ..

# ---------------------------------------------------------
# Cleanup
# ---------------------------------------------------------
cd ..
rm -rf $TMP_DIR

echo "✅ All binaries installed successfully to $VENV_BIN!"
echo "You can now use them directly via 'uv run', e.g., 'uv run mmseqs -h'"
