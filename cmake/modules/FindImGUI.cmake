# Provides the ImGUI_target library.
#
# imgui itself (docking branch) now comes from Conan (see conanfile.txt) instead
# of FetchContent. Conan ships only the compiled core in libimgui.a; the GLFW +
# OpenGL3 backends and the std::string helper are shipped as *source* under the
# package's res/ folder and must be compiled into our own target here.

find_package(imgui REQUIRED) # Conan CMakeDeps -> imgui::imgui (headers + libimgui.a)
find_package(glfw3 REQUIRED) # Conan CMakeDeps -> glfw
find_package(OpenGL REQUIRED)

if (NOT TARGET ImGUI_target)

	# Locate the bindings that Conan drops under <pkg>/res. The imgui include dir
	# is <pkg>/include, so res/ sits one level up. imgui_RES_DIRS is empty in the
	# generated files, so we derive it from imgui_INCLUDE_DIRS (a plain path set by
	# CMakeDeps; the target's INTERFACE_INCLUDE_DIRECTORIES is a genex and unusable
	# for path math).
	list(GET imgui_INCLUDE_DIRS 0 _imgui_inc)
	get_filename_component(_imgui_root "${_imgui_inc}" DIRECTORY)
	set(_imgui_bindings "${_imgui_root}/res/bindings")
	set(_imgui_stdlib   "${_imgui_root}/res/misc/cpp")

	add_compile_definitions(IMGUI_DEFINE_MATH_OPERATORS)

	add_library(ImGUI_target
		${_imgui_bindings}/imgui_impl_glfw.cpp
		${_imgui_bindings}/imgui_impl_opengl3.cpp
		${_imgui_stdlib}/imgui_stdlib.cpp
	)

	# imgui::imgui already propagates the imgui headers publicly, so consumers of
	# ImGUI_target get imgui.h / imgui_internal.h transitively.
	target_link_libraries(ImGUI_target
	PUBLIC
		imgui::imgui
		glfw
		OpenGL::GL
	)

	target_include_directories(ImGUI_target
	PUBLIC
		${_imgui_bindings}
		${_imgui_stdlib}
	)

endif()

set(ImGUI_INCLUDE_DIR ${_imgui_bindings} ${_imgui_stdlib})
set(ImGUI_FOUND TRUE)
set(ImGUI_LIBRARIES "ImGUI_target")
