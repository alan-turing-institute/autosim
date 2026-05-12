#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "This installer is for macOS only." >&2
    exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew is required before running this script." >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required before running this script." >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Native dependencies
brew install open-mpi fftw llvm libomp pkg-config

# hdf5-mpi conflicts with regular hdf5 if hdf5 is linked
if brew list --versions hdf5 >/dev/null 2>&1; then
    brew unlink hdf5 || true
fi
brew install hdf5-mpi
brew link hdf5-mpi || true

# Base autosim/dev env
uv sync --extra dev

# Dedalus build environment
export MPI_PATH="$(brew --prefix open-mpi)"
export FFTW_PATH="$(brew --prefix fftw)"
export HDF5_DIR="$(brew --prefix hdf5-mpi)"
export HDF5_MPI=ON

export PATH="$(brew --prefix llvm)/bin:$PATH"
export CC=mpicc
export MPICC=mpicc
export OMPI_CC="$(brew --prefix llvm)/bin/clang"
export CXX="$(brew --prefix llvm)/bin/clang++"

export CPPFLAGS="-I$(brew --prefix libomp)/include"
export LDFLAGS="-L$(brew --prefix libomp)/lib -Wl,-rpath,$(brew --prefix libomp)/lib"
export PKG_CONFIG_PATH="$(brew --prefix hdf5-mpi)/lib/pkgconfig:$(brew --prefix fftw)/lib/pkgconfig:$(brew --prefix open-mpi)/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# Build Python native bindings against Homebrew MPI/HDF5
uv pip install -U cython numpy setuptools wheel pkgconfig
uv pip install --force-reinstall --no-cache --no-binary=mpi4py mpi4py
uv pip install --force-reinstall --no-cache --no-build-isolation --no-binary=h5py h5py

# Build Dedalus
uv pip install --no-cache --no-build-isolation dedalus==3.0.5

# Verify
uv run python -c "import dedalus.public as d3; print('dedalus ok')"
uv run python -c "from autosim.experimental.simulations import ShallowWaterSphereDedalus; print('autosim spherical ok')"
