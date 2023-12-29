if(${USE_GUI})

	set(OpenGL_GL_PREFERENCE GLVND)
	cmake_policy(SET CMP0079 NEW)
endif()

add_subdirectory(CVolume/)
