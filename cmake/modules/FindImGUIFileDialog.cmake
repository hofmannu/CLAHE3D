include(FetchContent)

find_package(ImGUI REQUIRED)

FetchContent_Declare(
  ImGuiFileDialog
  GIT_REPOSITORY https://github.com/aiekick/ImGuiFileDialog
  GIT_TAG ee77f190861193a995435c388682f1512f6fd29a
)
FetchContent_GetProperties(ImGuiFileDialog)
if (NOT ImGuiFileDialog_POPULATED)
  FetchContent_MakeAvailable(ImGuiFileDialog)
  add_library(ImGuiFileDialog_target
    ${imguifiledialog_SOURCE_DIR}/ImGuiFileDialog.cpp
    )

  target_link_libraries(ImGuiFileDialog_target PUBLIC 
    ${ImGUI_LIBRARIES})
	
  target_include_directories(ImGuiFileDialog_target PUBLIC
    ${imgui_SOURCE_DIR}
    ${imguifiledialog_SOURCE_DIR}
  )
endif()

SET(ImGUIFileDialog_INCLUDE_DIR "${imguifiledialog_SOURCE_DIR}")
SET(ImGUIFileDialog_LIBRARIES ImGuiFileDialog_target)
