# little cmake module to get hands on catch2
# todo add option to use global installation candidate
include(FetchContent)

FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.13.5 # or a later release
)

FetchContent_GetProperties(catch2)
if(NOT catch2_POPULATED)
  FetchContent_MakeAvailable(catch2)
  add_subdirectory(${catch2_SOURCE_DIR} ${catch2_BINARY_DIR})
endif()

set(catch2_INCLUDE_DIR "${catch2_SOURCE_DIR}/;${catch2_SOURCE_DIR}/src/;" CACHE PATH "Path to include folder for Catch2")
set(catch2_LIBRARIES "Catch2::Catch2" CACHE STRING "Library for Catch2")
set(catch2_FOUND TRUE CACHE BOOL "Catch2 found")
