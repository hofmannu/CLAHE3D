cmake_minimum_required(VERSION 3.12)

set(USE_CUDA FALSE) # flag to enable and disable GPU support
set(USE_GUI TRUE) # flag to enable GUI building

if(${USE_CUDA})
	message("Compiling with GPU support")
	project(clahe3d VERSION 0.1
		DESCRIPTION "Contrast limited adaptive histogram equilization in 3D"
		LANGUAGES CUDA CXX)
	add_compile_definitions(USE_CUDA)
else()
	message("Compiling wihtout GPU support")
	project(clahe3d VERSION 0.1
		DESCRIPTION "Contrast limited adaptive histogram equilization in 3D"
		LANGUAGES CXX)
endif()


if(${USE_CUDA})
	enable_language(CUDA)
	find_package(CUDA REQUIRED QUIET)
	INCLUDE(FindCUDA)
	INCLUDE_DIRECTORIES(/opt/cuda/include include)

	set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
	set(BUILD_SHARED_LIBS OFF)

	set(CMAKE_EXPORT_COMPILE_COMMANDS ON )
	# set(CMAKE_CUDA_FLAGS "${CUDA_CUDA_FLAGS} -rdc=true -ftz=false -prec-div=true -prec-sqrt=true")
	set(CMAKE_CUDA_FLAGS "${CUDA_CUDA_FLAGS} -rdc=true -Xptxas -v -O2")

	set(
	    CUDA_NVCC_FLAGS
			${CUDA_NVCC_FLAGS};
			-x cu -rdc=true -ftz=false -prec-div=true -prec-sqrt=true -g
	)

	# please adapt this to the correct architecture for your device
	SET(CMAKE_CUDA_ARCHITECTURES 75)

endif()

add_subdirectory(lib/)
add_subdirectory(src/)
add_subdirectory(utests/)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# add some flags for profiling
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg") # for profiling of the code using gprof
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

add_executable(main_exp src/main.cpp)
target_link_libraries(main_exp PUBLIC Histeq)
install(TARGETS main_exp DESTINATION bin)

add_executable(clahe3d_scripted src/clahe3d_scripted.cpp)
target_link_libraries(clahe3d_scripted PUBLIC Histeq Volume) 
install(TARGETS clahe3d_scripted DESTINATION bin)

# gui specific stuff
if(${USE_GUI})
	set(OpenGL_GL_PREFERENCE GLVND)
	cmake_policy(SET CMP0079 NEW)
	add_executable(main_gui src/main_gui.cpp)
	target_link_libraries(main_gui PUBLIC Gui)
endif()

