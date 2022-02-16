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

add_executable(UtestVector3 utest_vector3.cpp)

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

add_library(Nifti nifti.cpp)

add_library(Histogram histogram.cpp)


if(${USE_GUI})
	add_library(ImGuiFileDialog ImGuiFileDialog.cpp)
	target_link_libraries(ImGuiFileDialog PUBLIC ImGui)

	add_library(ImGuiPlot imgui_plot.cpp)
	add_library(ColorMapper color_mapper.cpp)

	set(OpenGL_GL_PREFERENCE GLVND)
	cmake_policy(SET CMP0079 NEW)

	add_library(ImGuiImplOpenGL3 imgui_impl_opengl3.cpp)
	target_link_libraries(ImGuiImplOpenGL3 PUBLIC ImGui OpenGL)

	find_package(SDL2 REQUIRED)
	add_library(ImGuiImplSdl imgui_impl_sdl.cpp)
	target_link_libraries(ImGuiImplSdl PUBLIC ImGui SDL2 OpenGL)

	find_package(OpenGL REQUIRED)
	include_directories(${OPENGL_INCLUDE_DIRS})

	find_package(glfw3 REQUIRED)
	include_directories(${GLFW3_INCLUDE_DIR})
	link_libraries(${GLFW_LIBRARY_DIRS})

	add_library(Slicer slicer.cpp)


	add_library(Gui gui.cpp)
	target_link_libraries(Gui PUBLIC
		ImGuiFileDialog
		ImGuiImplOpenGL3
		GLEW
		ImGuiImplSdl
		OpenGL::GL
		${CMAKE_DL_LIBS}
		Histeq
		Nifti
		Slicer
		ColorMapper
		Histogram
		ImGuiPlot
		)
	set_property(TARGET Gui PROPERTY CUDA_SEPARABLE_COMPILATION ON)
	
endif()