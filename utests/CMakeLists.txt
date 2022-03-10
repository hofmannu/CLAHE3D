# subfolder containing all the important unit tests

# declaration of unit test executables
add_executable(UtestProc utest_proc.cpp)
target_link_libraries(UtestProc PUBLIC Histeq)

add_executable(UtestOverwrite utest_overwrite.cpp)
target_link_libraries(UtestOverwrite PUBLIC Histeq)

add_executable(UtestNoiseLevel utest_noiseLevel.cpp)
target_link_libraries(UtestNoiseLevel PUBLIC Histeq)

add_executable(UtestBinarization utest_binarization.cpp)
target_link_libraries(UtestBinarization PUBLIC Histeq)

add_executable(UtestVector3 utest_vector3.cpp)

add_executable(UtestGenfilt utest_genfilt.cpp)
target_link_libraries(UtestGenfilt PUBLIC Genfilt)

add_executable(UtestMeanfilt utest_meanfilt.cpp)
target_link_libraries(UtestMeanfilt PUBLIC Meanfilt)

add_executable(UtestGaussfilt utest_gaussfilt.cpp)
target_link_libraries(UtestGaussfilt PUBLIC Gaussfilt)

add_executable(UtestMedianfilt utest_medianfilt.cpp)
target_link_libraries(UtestMedianfilt PUBLIC Medianfilt)

add_executable(UtestNormalizer utest_normalizer.cpp)
target_link_libraries(UtestNormalizer PUBLIC Normalizer)

# cuda specific test functions
if(${USE_CUDA})

	add_executable(UtestProcGpu utest_proc_gpu.cpp)
	target_link_libraries(UtestProcGpu PUBLIC Histeq)
	set_property(TARGET UtestProcGpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)

	add_executable(UtestCdfGpu utest_cdf_gpu.cpp)
	target_link_libraries(UtestCdfGpu PUBLIC Histeq)
	set_property(TARGET UtestCdfGpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)

	add_executable(UtestEqGpu utest_eq_gpu.cpp)
	target_link_libraries(UtestEqGpu PUBLIC Histeq)
	set_property(TARGET UtestEqGpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)

	add_executable(UtestGpuFull utest_gpu_full.cpp)
	target_link_libraries(UtestGpuFull PUBLIC Histeq)
	set_property(TARGET UtestGpuFull PROPERTY CUDA_SEPARABLE_COMPILATION ON)

endif()

