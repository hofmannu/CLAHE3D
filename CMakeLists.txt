cmake_minimum_required(VERSION 3.12)

project(clahe3d 
	VERSION 0.2
	DESCRIPTION "Contrast limited adaptive histogram equilization in 3D"
  LANGUAGES CXX C
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(USE_CUDA TRUE) # flag to enable and disable GPU support
set(USE_GUI TRUE) # flag to enable GUI building

if(${USE_CUDA})
  message(STATUS "Compiling with CUDA support")
	add_compile_definitions(USE_CUDA)
	enable_language(CUDA)
	find_package(CUDA REQUIRED)
	INCLUDE_DIRECTORIES(/opt/cuda/include include)

	set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
	set(BUILD_SHARED_LIBS OFF)

	set(CMAKE_EXPORT_COMPILE_COMMANDS ON )
	set(CMAKE_CUDA_FLAGS "${CUDA_CUDA_FLAGS} -rdc=true -Xptxas -v -O2")

	set(
	    CUDA_NVCC_FLAGS
			${CUDA_NVCC_FLAGS};
			-x cu -rdc=true -ftz=false -prec-div=true -prec-sqrt=true -g
	)

	# please adapt this to the correct architecture for your device
  SET(CMAKE_CUDA_ARCHITECTURES 75 86)
else()
  message(STATUS "Compiling without CUDA support")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules")
find_package(ImGUI REQUIRED)
find_package(ImGUIFileDialog REQUIRED)
find_package(ImPlot REQUIRED)
find_package(Catch2 REQUIRED)

add_subdirectory(lib/)
add_subdirectory(src/)
add_subdirectory(utests/)

# add some flags for profiling
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pg -lpthread") # for profiling of the code using gprof
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

# small exec to run clahe3d from command line
add_executable(clahe3d_scripted src/clahe3d_scripted.cpp)
target_link_libraries(clahe3d_scripted PUBLIC Histeq Volume) 
install(TARGETS clahe3d_scripted DESTINATION bin)

# small exec to run histogram analysis from command line
add_executable(histogram_scripted src/histogram_scripted.cpp)
target_link_libraries(histogram_scripted PUBLIC Histogram Volume) 
install(TARGETS histogram_scripted DESTINATION bin)

# gui specific stuff
if(${USE_GUI})
	set(OpenGL_GL_PREFERENCE GLVND)
	cmake_policy(SET CMP0079 NEW)
	add_executable(main_gui src/main_gui.cpp)
	target_link_libraries(main_gui PUBLIC Gui)
endif()


enable_testing() # this cannot be in subdirectory!

# Register tests with CTest
add_test(NAME all_tests COMMAND AllTests)

if(${USE_CUDA})
	add_test(NAME all_tests_gpu COMMAND AllTestsGpu)
endif()
