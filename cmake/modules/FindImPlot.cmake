include(FetchContent)

find_package(ImGUI)

FetchContent_Declare(
  ImPlot
  GIT_REPOSITORY https://github.com/epezent/implot
  GIT_TAG v0.16
)
FetchContent_GetProperties(ImPlot)
if (NOT ImPlot_POPULATED)
  FetchContent_Populate(ImPlot)
  add_library(ImPlot_target
    ${implot_SOURCE_DIR}/implot.cpp
    ${implot_SOURCE_DIR}/implot_items.cpp

    )

  target_link_libraries(ImPlot_target PUBLIC 
    ${ImGUI_LIBRARIES})
	
  target_include_directories(ImPlot_target PUBLIC
    ${imgui_SOURCE_DIR}
    ${implot_SOURCE_DIR}
  )
endif()

SET(ImPlot_INCLUDE_DIR "${implot_SOURCE_DIR}/")
SET(ImPlot_LIBRARIES ImPlot_target)