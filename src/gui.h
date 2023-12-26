/* 
	a little graphical user interface building on our volume processing toolbox
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 03.03.2022
*/

#ifndef GUI_H
#define GUI_H

#include <SDL2/SDL.h>
#include <GL/glew.h>

#include "../lib/imgui/imgui.h"
#include "../lib/imgui/misc/cpp/imgui_stdlib.h"

// all the imgui stuff goes here
#include "imgui_impl_sdl.h"
#include "imgui_impl_opengl3.h"
#include "ImGuiFileDialog.h"
#include "imgui_plot.h"

#include "volproc.h"
#include "histogram.h"
#include "../lib/CVolume/src/volume.h"
#include "color_mapper.h"

#include "slicer.h"
#include "log.h"

class gui
{
private:
	const char* windowTitle = "Volume Processing Toolbox";
	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 0.10f); // bg color
	
	// image processing parts
	volproc proc;
	volume* inputVol;
	volume* outputVol;
	const histogram* inputHist;
	const histogram* outputHist;

	void MainDisplayCode(); // iterates over the main display boxes 
	void DataLoaderWindow(); 
	void SettingsWindow();
	void SlicerWindow();
	void Console();
	void ExportData();

	bool showRaw = true;
	bool flagGpu = true; // should we process on the GPU

	// helper function to display stuff
	void ImImagesc(
		const float* data, const uint64_t sizex, const uint64_t sizey, 
		GLuint* out_texture, const color_mapper myCMap);
	
	slicer mySlice; // used for preview of processed and unprocessed dataset

	// raw viz
	GLuint sliceZ; 
	GLuint sliceX; 
	GLuint sliceY; 

	// colomaps
	color_mapper rawMap; // for raw data vizualization
	color_mapper procMap; // for processed data vizualization

	// all settings stucts for different procedures go here
	meanfiltsett sett_meanfilt;
	medianfiltsett sett_medianfilt;
	gaussfiltsett sett_gaussfilt;
	thresholdersett sett_thresholdfilt;
	histeqsett sett_histeq;
	normalizersett<float> sett_normalizer;

	std::string outputPath;
	std::string outputFile;

	bool showLog = true;
	bool showWarnings = true;
	bool showErrors = true;

	// helper functions
	void check_mapLimits();

public:
	gui();
	~gui();
	void InitWindow(int *argcp, char**argv);
};

#endif