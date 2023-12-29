# prepare for cuda compilation
if (${USE_CUDA})
	find_package(CUDA REQUIRED)
	INCLUDE_DIRECTORIES(/opt/cuda/include include)

	set(CUDA_SEPARABLE_COMPILATION ON)
	set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
	set(BUILD_SHARED_LIBS OFF)
	set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
	set(CMAKE_CUDA_FLAGS "${CUDA_CUDA_FLAGS} -rdc=true --ptxas-options=-v -O2")
endif()

add_library(Gridder gridder.cpp)

add_library(Histeq histeq.cpp)
if (${USE_CUDA})
	set_property(TARGET Histeq PROPERTY CUDA_SEPARABLE_COMPILATION ON)
	target_link_libraries(Histeq PUBLIC Gridder CudaTools ${CUDA_LIBRARY})
	set_source_files_properties(histeq.cpp PROPERTIES LANGUAGE CUDA)
else()
	target_link_libraries(Histeq PUBLIC Gridder)
endif()

if (${USE_CUDA})
	add_library(CudaTools cudaTools.cu)
endif()

# general spatial domain filter class and all its variants (kernel convolutions)
add_library(Genfilt genfilt.cpp)
if (${USE_CUDA})
	target_link_libraries(Genfilt PUBLIC CudaTools ${CUDA_LIBRARY})
	set_property(TARGET Genfilt PROPERTY CUDA_SEPARABLE_COMPILATION ON)
	set_source_files_properties(genfilt.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_library(Meanfilt meanfilt.cpp)
target_link_libraries(Meanfilt PUBLIC Genfilt)
if (${USE_CUDA})
	target_link_libraries(Meanfilt PUBLIC CudaTools ${CUDA_LIBRARY})
	set_property(TARGET Meanfilt PROPERTY CUDA_SEPARABLE_COMPILATION ON)
	set_source_files_properties(meanfilt.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_library(Gaussfilt gaussfilt.cpp)
target_link_libraries(Gaussfilt PUBLIC Genfilt)
if (${USE_CUDA})
	target_link_libraries(Gaussfilt PUBLIC CudaTools ${CUDA_LIBRARY})
	set_property(TARGET Gaussfilt PROPERTY CUDA_SEPARABLE_COMPILATION ON)
	set_source_files_properties(gaussfilt.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_library(Medianfilt medianfilt.cpp)
target_link_libraries(Medianfilt PUBLIC Genfilt)
if (${USE_CUDA})
	target_link_libraries(Medianfilt PUBLIC CudaTools ${CUDA_LIBRARY})
	set_property(TARGET Medianfilt PROPERTY CUDA_SEPARABLE_COMPILATION ON)
	set_source_files_properties(medianfilt.cpp PROPERTIES LANGUAGE CUDA)
endif()

# simple function like thresholding
add_library(Thresholder thresholder.cpp)

add_library(Histogram histogram.cpp)

# generating log files and console like output with timestamps
add_library(Log log.cpp)

add_library(Normalizer normalizer.cpp)

# overall volume processing class

add_library(Volproc volproc.cpp)
if (${USE_CUDA})
	set_source_files_properties(volproc.cpp PROPERTIES LANGUAGE CUDA)
	set_property(TARGET Volproc PROPERTY CUDA_SEPARABLE_COMPILATION ON)
	# set_source_files_properties(volproc.cpp PROPERTIES LANGUAGE CUDA)
endif()
target_link_libraries(Volproc PUBLIC
	Genfilt
	Meanfilt
	Gaussfilt
	Medianfilt
	Thresholder
	Histeq
	Histogram
	Normalizer
	Log)

add_library(Lexer lexer.cpp)

if(${USE_GUI})

	add_library(ColorMapper color_mapper.cpp)

	add_library(Slicer slicer.cpp)

	add_library(Gui gui.cpp)
	target_link_libraries(Gui PUBLIC
		${ImGUIFileDialog_LIBRARIES}
		${ImGUI_LIBRARIES}
		${CMAKE_DL_LIBS}
		${glfw3_LIBRARIES}
		Histeq
		Volume
		Volproc
		ColorMapper
		${ImPlot_LIBRARIES}
		Slicer
		)
	target_include_directories(Gui PUBLIC
		${ImPlot_INCLUDE_DIR}
		${ImGUIFileDialog_INCLUDE_DIR}
		${ImGUI_INCLUDE_DIR}
		)
	set_property(TARGET Gui PROPERTY CUDA_SEPARABLE_COMPILATION ON)
	
endif()