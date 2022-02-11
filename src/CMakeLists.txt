add_library(Histeq histeq.cpp)
target_link_libraries(Histeq PUBLIC Interpgrid)

add_library(Interpgrid interpGrid.cpp)