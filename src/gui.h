#ifndef GUI_H
#define GUI_H


#include <SDL2/SDL.h>
#include <GL/glew.h>    // Initialize with gl3wInit()

#include "imgui_impl_sdl.h"
#include "imgui_impl_opengl3.h"
#include "ImGuiFileDialog.h"

#include "histeq.h"
#include "nifti.h"
#include "slicer.h"
#include "color_mapper.h"

class gui
{
private:
	const char* windowTitle = "CLAHE3d";
	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 0.10f); // bg color
	
	histeq histoEq;
	nifti niiReader;
	slicer mySlice;

	void MainDisplayCode(); // iterates over the main display boxes 
	void DataLoaderWindow(); 
	void SettingsWindow();
	void SlicerWindow();

	bool isDataLoaded = 0;
	bool flagGpu = 1; // should we process on the GPU

	// helper function to display stuff
	void ImImagesc(
		const float* data, const uint64_t sizex, const uint64_t sizey, 
		GLuint* out_texture, const color_mapper myCMap);

	GLuint rawSliceZ; 
	color_mapper rawMap;

public:
	gui();
	void InitWindow(int *argcp, char**argv);
};

#endif

// itk::NiftiImageIO::Pointer nifti = itk::NiftiImageIO::New();
// ImageTypeReader::Pointer fileReader = ImageTypeReader::New();
// fileReader->SetImageIO( nifti );
// fileReader->SetFileName( file.string() );