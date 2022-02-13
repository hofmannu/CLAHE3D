# prepare for cuda compilation
if (${USE_CUDA})
	set(CUDA_SEPARABLE_COMPILATION ON)
	set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
	set(BUILD_SHARED_LIBS OFF)
	set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
	# set(CMAKE_CUDA_FLAGS "${CUDA_CUDA_FLAGS} -rdc=true -ftz=false -prec-div=true -prec-sqrt=true")
	set(CMAKE_CUDA_FLAGS "${CUDA_CUDA_FLAGS} -rdc=true --ptxas-options=-v -O2")
endif()

add_library(Gridder gridder.cpp)

add_library(Histeq histeq.cpp)
if (${USE_CUDA})
	set_property(TARGET Histeq PROPERTY CUDA_ARCHITECTURES 50 75)
	set_property(TARGET Histeq PROPERTY CUDA_SEPARABLE_COMPILATION ON)
	target_link_libraries(Histeq PUBLIC Gridder CudaTools)
	set_source_files_properties(histeq.cpp PROPERTIES LANGUAGE CUDA)
else()
	target_link_libraries(Histeq PUBLIC Gridder)
endif()

if (${USE_CUDA})
	add_library(CudaTools cudaTools.cu)
endif()

# set_property(TARGET CudaTools PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# set_property(TARGET CudaTools PROPERTY CUDA_ARCHITECTURES 50 75)

# declaration of unit test executables
add_executable(UtestProc utest_proc.cpp)
target_link_libraries(UtestProc PUBLIC Histeq)

add_executable(UtestOverwrite utest_overwrite.cpp)
target_link_libraries(UtestOverwrite PUBLIC Histeq)

add_executable(UtestNoiseLevel utest_noiseLevel.cpp)
target_link_libraries(UtestNoiseLevel PUBLIC Histeq)

add_executable(UtestBinarization utest_binarization.cpp)
target_link_libraries(UtestBinarization PUBLIC Histeq)

# cuda specific test functions
if(${USE_CUDA})

	add_executable(UtestProcGpu utest_proc_gpu.cpp)
	target_link_libraries(UtestProcGpu PUBLIC Histeq)

	add_executable(UtestCdfGpu utest_cdf_gpu.cpp)
	target_link_libraries(UtestCdfGpu PUBLIC Histeq)

	add_executable(UtestEqGpu utest_eq_gpu.cpp)
	target_link_libraries(UtestEqGpu PUBLIC Histeq)

endif()
