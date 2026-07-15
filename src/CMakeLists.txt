# prepare for cuda compilation
if (${USE_CUDA})
	find_package(CUDA REQUIRED)
	INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS} include)

	set(CUDA_SEPARABLE_COMPILATION ON)
	set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
	set(BUILD_SHARED_LIBS OFF)
	set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
	set(CMAKE_CUDA_FLAGS "${CUDA_CUDA_FLAGS} -rdc=true --ptxas-options=-v -O2")
endif()

# clahe3d: the single core library
set(CLAHE3D_SOURCES
	gridder.cpp
	histeq.cpp
	genfilt.cpp
	meanfilt.cpp
	gaussfilt.cpp
	medianfilt.cpp
	thresholder.cpp
	histogram.cpp
	log.cpp
	normalizer.cpp
	volproc.cpp
	lexer.cpp
)

if (${USE_CUDA})
	list(APPEND CLAHE3D_SOURCES cudaTools.cu)
endif()

add_library(clahe3d ${CLAHE3D_SOURCES})

# volproc.h -> volume.h needs the HDF5 include path, which now propagates from
# Conan's hdf5::hdf5_cpp target through Volume.
target_link_libraries(clahe3d PUBLIC Volume)

if (${USE_CUDA})
	# the translation units that actually contain / include device code have to
	# be compiled by nvcc; the rest stay plain C++ within the same target.
	set_source_files_properties(
		histeq.cpp
		genfilt.cpp
		meanfilt.cpp
		gaussfilt.cpp
		medianfilt.cpp
		volproc.cpp
		PROPERTIES LANGUAGE CUDA)
	set_property(TARGET clahe3d PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

# Gui: optional graphical front-end (only built with USE_GUI)
if(${USE_GUI})

	add_library(Gui
		gui.cpp
		color_mapper.cpp
		slicer.cpp
	)
	target_link_libraries(Gui PUBLIC
		clahe3d
		Volume
		${ImGUIFileDialog_LIBRARIES}
		${ImGUI_LIBRARIES}
		${CMAKE_DL_LIBS}
		${glfw3_LIBRARIES}
		${ImPlot_LIBRARIES}
	)
	target_include_directories(Gui PUBLIC
		${ImPlot_INCLUDE_DIR}
		${ImGUIFileDialog_INCLUDE_DIR}
		${ImGUI_INCLUDE_DIR}
	)
	if (${USE_CUDA})
		set_property(TARGET Gui PROPERTY CUDA_SEPARABLE_COMPILATION ON)
	endif()

endif()
