#!/usr/bin/env bash
#
# Quick build helper for CLAHE3D.
#
#   ./build.sh            # Debug build (default)
#   ./build.sh Debug
#   ./build.sh Release
#
# It resolves dependencies with Conan, configures against the generated CMake
# toolchain, and builds into a folder named after the build type.
set -euo pipefail

BUILD_TYPE="${1:-Debug}"

if [[ "$BUILD_TYPE" != "Release" && "$BUILD_TYPE" != "Debug" ]]; then
	echo "Usage: $0 [Release|Debug]  (got '$BUILD_TYPE')" >&2
	exit 1
fi

# Run from the project root regardless of the caller's working directory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

BUILD_DIR="$BUILD_TYPE"

# 1. Resolve / build third-party dependencies and emit conan_toolchain.cmake.
#    Pin to conancenter: all deps live there, and it avoids interactive auth
#    prompts from other configured remotes (e.g. a private gitea).
conan install . \
	--output-folder="$BUILD_DIR" \
	--build=missing \
	-r conancenter \
	-s build_type="$BUILD_TYPE" \
	-s compiler.cppstd=20

# 2. Configure against the Conan toolchain (path is relative to the build dir).
cmake -S . -B "$BUILD_DIR" \
	-DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake \
	-DCMAKE_BUILD_TYPE="$BUILD_TYPE"

# 3. Build everything.
cmake --build "$BUILD_DIR" -j"$(nproc)"

echo
echo "Done. Binaries are in $BUILD_DIR/ (run 'ctest' from there to test)."
