# Unit Tests with Catch2

This directory contains unit tests for the CLAHE3D project, structured using the [Catch2](https://github.com/catchorg/Catch2) testing framework (v2.13.5).

## Test Structure

### Test Organization

All tests are organized into two main executables:

1. **AllTests** - Contains all CPU-based tests
2. **AllTestsGpu** - Contains all GPU-based tests (built only when `USE_CUDA=TRUE`)

### Test Files

#### CPU Tests
- `test_main.cpp` - Main test runner with Catch2 entry point
- `utest_vector3.cpp` - Vector3 class operations
- `utest_padding.cpp` - Volume padding functionality
- `utest_proc.cpp` - Histogram equalization processing
- `utest_overwrite.cpp` - Overwrite flag functionality
- `utest_noiseLevel.cpp` - Noise level handling
- `utest_binarization.cpp` - Binarization operations
- `utest_genfilt.cpp` - Generic filter operations
- `utest_meanfilt.cpp` - Mean filter operations
- `utest_gaussfilt.cpp` - Gaussian filter operations
- `utest_medianfilt.cpp` - Median filter operations
- `utest_normalizer.cpp` - Array normalization
- `utest_lexer.cpp` - Lexer parsing

#### GPU Tests (CUDA Required)
- `utest_proc_gpu.cpp` - GPU histogram equalization
- `utest_cdf_gpu.cpp` - CPU/GPU CDF comparison
- `utest_eq_gpu.cpp` - CPU/GPU equalization comparison
- `utest_gpu_full.cpp` - Full GPU pipeline
- `utest_medianfilt_gpu.cpp` - GPU median filter
- `utest_genfilt_gpu.cpp` - GPU generic filter
- `utest_cuda_tools.cpp` - CUDA tools and device properties

## Building the Tests

### Prerequisites
- CMake 3.12 or higher
- C++20 compatible compiler
- HDF5 library (for CVolume dependency)
- CUDA Toolkit (optional, for GPU tests)

### Build Commands

```bash
mkdir build
cd build
cmake ..
make AllTests        # Build CPU tests
make AllTestsGpu     # Build GPU tests (if CUDA enabled)
```

### Running Tests

Run all tests:
```bash
./utests/AllTests
```

Run specific test by name:
```bash
./utests/AllTests "vector3 operations"
```

Run tests with verbose output:
```bash
./utests/AllTests -s
```

Run tests with compact reporter:
```bash
./utests/AllTests --reporter compact
```

List all available tests:
```bash
./utests/AllTests --list-tests
```

### Using CTest

The tests are also registered with CTest:
```bash
ctest                    # Run all tests
ctest -R vector3         # Run tests matching pattern
ctest -V                 # Verbose output
```

## Test Tags

Tests are organized with tags for filtering:

- `[vector3]` - Vector3 class tests
- `[histeq]` - Histogram equalization tests
- `[genfilt]`, `[meanfilt]`, `[gaussfilt]`, `[medianfilt]` - Filter tests
- `[normalizer]` - Normalization tests
- `[lexer]` - Lexer tests
- `[gpu]` - GPU-specific tests
- `[padding]` - Padding tests

Run tests by tag:
```bash
./utests/AllTests [vector3]
./utests/AllTests [histeq]
```

## Catch2 Features Used

- **TEST_CASE**: Main test declaration with descriptive names and tags
- **SECTION**: Logical grouping within test cases
- **REQUIRE**: Assertion that stops test on failure
- **INFO**: Contextual information printed on test failure

## Migration from Old Test Structure

The tests were previously structured as standalone executables with manual main() functions and printf/throw error handling. They have been migrated to use Catch2 with the following improvements:

1. **Single Test Executable**: All tests compile into one executable instead of 19+ separate binaries
2. **Better Assertions**: Using REQUIRE/INFO instead of if/printf/throw
3. **Test Discovery**: Automatic test registration and discovery
4. **Better Output**: Structured test output with pass/fail reporting
5. **Selective Running**: Run specific tests by name or tag
6. **Integration**: Native CTest integration for CI/CD pipelines

## Continuous Integration

The test framework is designed to work well with CI systems:

- Exit code 0 on all tests passing
- Exit code 1 on any test failure
- Structured output for parsing
- Support for XML/JSON reporters for CI integration
