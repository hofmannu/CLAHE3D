# subfolder containing all the important unit tests

# Create a single test executable with all test files
set(TEST_SOURCES
	test_main.cpp
	utest_vector3.cpp
	utest_padding.cpp
	utest_proc.cpp
	utest_overwrite.cpp
	utest_noiseLevel.cpp
	utest_binarization.cpp
	utest_genfilt.cpp
	utest_meanfilt.cpp
	utest_gaussfilt.cpp
	utest_medianfilt.cpp
	utest_normalizer.cpp
	utest_lexer.cpp
)

add_executable(AllTests ${TEST_SOURCES})
target_link_libraries(AllTests PRIVATE 
	Catch2::Catch2
	Histeq
	Genfilt
	Meanfilt
	Gaussfilt
	Medianfilt
	Normalizer
	Lexer
)

# cuda specific test functions
if(${USE_CUDA})
	set(CUDA_TEST_SOURCES
		test_main.cpp
		utest_proc_gpu.cpp
		utest_cdf_gpu.cpp
		utest_eq_gpu.cpp
		utest_medianfilt_gpu.cpp
		utest_genfilt_gpu.cpp
		utest_gpu_full.cpp
		utest_cuda_tools.cpp
	)

	add_executable(AllTestsGpu ${CUDA_TEST_SOURCES})
	target_link_libraries(AllTestsGpu PRIVATE
		Catch2::Catch2
		Histeq
		Medianfilt
		Genfilt
		CudaTools
	)
	set_property(TARGET AllTestsGpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

