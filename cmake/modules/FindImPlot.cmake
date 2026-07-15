# ImPlot is kept on FetchContent (compiled from source) on purpose: ConanCenter's
# prebuilt implot is built against the non-docking imgui and would ABI-mismatch our
# docking imgui at runtime. Compiling it here against ImGUI_target (the Conan
# docking imgui) keeps ImGuiContext layout consistent.
include(FetchContent)

find_package(ImGUI REQUIRED MODULE) # our module, not Conan's imgui-config

FetchContent_Declare(
  ImPlot
  GIT_REPOSITORY https://github.com/epezent/implot
  GIT_TAG v0.17
)
FetchContent_GetProperties(ImPlot)
if (NOT implot_POPULATED)
  FetchContent_MakeAvailable(ImPlot)
endif()

if (NOT TARGET ImPlot_target)
  add_library(ImPlot_target
    ${implot_SOURCE_DIR}/implot.cpp
    ${implot_SOURCE_DIR}/implot_items.cpp
  )

  # ImGUI_target propagates the Conan imgui include dir, so implot.cpp finds imgui.h
  target_link_libraries(ImPlot_target PUBLIC
    ${ImGUI_LIBRARIES})

  target_include_directories(ImPlot_target PUBLIC
    ${implot_SOURCE_DIR}
  )
endif()

SET(ImPlot_INCLUDE_DIR "${implot_SOURCE_DIR}/")
SET(ImPlot_LIBRARIES ImPlot_target)
