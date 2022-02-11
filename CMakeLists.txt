cmake_minimum_required(VERSION 3.12)


project(clahe3d VERSION 0.1
	DESCRIPTION "Contrast limited adaptive histogram equilization in 3D"
	LANGUAGES CXX)

add_subdirectory(src/)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

add_executable(main_exp src/main.cpp)
target_link_libraries(main_exp PUBLIC Histeq Interpgrid)

install(TARGETS main_exp DESTINATION bin)
