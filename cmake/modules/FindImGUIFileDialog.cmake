# ImGuiFileDialog is not on ConanCenter, so it stays on FetchContent. It only needs
# imgui.h on the include path, which ImGUI_target (Conan docking imgui) provides
# transitively.
include(FetchContent)

find_package(ImGUI REQUIRED MODULE) # our module, not Conan's imgui-config

FetchContent_Declare(
  ImGuiFileDialog
  GIT_REPOSITORY https://github.com/aiekick/ImGuiFileDialog
  GIT_TAG ee77f190861193a995435c388682f1512f6fd29a
)
FetchContent_GetProperties(ImGuiFileDialog)
if (NOT imguifiledialog_POPULATED)
  FetchContent_MakeAvailable(ImGuiFileDialog)
endif()

if (NOT TARGET ImGuiFileDialog_target)
  add_library(ImGuiFileDialog_target
    ${imguifiledialog_SOURCE_DIR}/ImGuiFileDialog.cpp
  )

  target_link_libraries(ImGuiFileDialog_target PUBLIC
    ${ImGUI_LIBRARIES})

  target_include_directories(ImGuiFileDialog_target PUBLIC
    ${imguifiledialog_SOURCE_DIR}
  )
endif()

SET(ImGUIFileDialog_INCLUDE_DIR "${imguifiledialog_SOURCE_DIR}")
SET(ImGUIFileDialog_LIBRARIES ImGuiFileDialog_target)
