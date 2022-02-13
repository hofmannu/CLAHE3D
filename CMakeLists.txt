cmake_minimum_required(VERSION 3.12)


project(clahe3d VERSION 0.1
	DESCRIPTION "Contrast limited adaptive histogram equilization in 3D"
	LANGUAGES CUDA CXX)

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

add_subdirectory(src/)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED true)

add_executable(main_exp src/main.cpp)
set_property(TARGET main_exp PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET main_exp PROPERTY CUDA_ARCHITECTURES 50 75)

target_link_libraries(main_exp PUBLIC Histeq Interpgrid)

install(TARGETS main_exp DESTINATION bin)
