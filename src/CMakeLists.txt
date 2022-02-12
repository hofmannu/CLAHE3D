# prepare for cuda compilation
enable_language(CUDA)
find_package(CUDA REQUIRED QUIET)
INCLUDE(FindCUDA)
INCLUDE_DIRECTORIES(/opt/cuda/include include)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_CUDA_FLAGS "${CUDA_CUDA_FLAGS} -rdc=true -ftz=false -prec-div=true -prec-sqrt=true")
set(CMAKE_CUDA_FLAGS "${CUDA_CUDA_FLAGS} -rdc=true --ptxas-options=-v -O2")

add_library(Histeq histeq.cu)
set_property(TARGET Histeq PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET Histeq PROPERTY CUDA_ARCHITECTURES 50 75)
target_link_libraries(Histeq PUBLIC Interpgrid)



add_library(Interpgrid interpGrid.cpp)