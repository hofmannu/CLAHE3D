# subfolder containing all the important unit tests
#
# All unit tests are compiled into a single Catch2 executable (Utests). Catch2
# provides main() via Catch2::Catch2WithMain, and every TEST_CASE is registered
# with CTest automatically through catch_discover_tests(), so there is no need to
# list the individual cases by hand in the top-level CMakeLists.txt.

# CPU unit tests (always built)
set(UTEST_SOURCES
	utest_proc.cpp
	utest_overwrite.cpp
	utest_noiseLevel.cpp
	utest_binarization.cpp
	utest_vector3.cpp
	utest_genfilt.cpp
	utest_padding.cpp
	utest_meanfilt.cpp
	utest_gaussfilt.cpp
	utest_medianfilt.cpp
	utest_normalizer.cpp
	utest_lexer.cpp
)

# GPU unit tests (only built with CUDA support)
if(${USE_CUDA})
	list(APPEND UTEST_SOURCES
		utest_proc_gpu.cpp
		utest_cdf_gpu.cpp
		utest_eq_gpu.cpp
		utest_medianfilt_gpu.cpp
		utest_genfilt_gpu.cpp
		utest_gpu_full.cpp
		utest_cuda_tools.cpp
	)
endif()

add_executable(Utests ${UTEST_SOURCES})
target_link_libraries(Utests PUBLIC
	Histeq
	Genfilt
	Meanfilt
	Gaussfilt
	Medianfilt
	Normalizer
	Lexer
	Catch2::Catch2WithMain
)

if(${USE_CUDA})
	target_link_libraries(Utests PUBLIC CudaTools)
	set_property(TARGET Utests PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

catch_discover_tests(Utests)
