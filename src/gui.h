/*
   a little graphical user interface building on our volume processing
   toolbox 
   Author: Urs Hofmann 
   Mail: mail@hofmannu.org 
   Date: 03.03.2022
*/

#ifndef GUI_H
#define GUI_H

#include "../lib/CVolume/src/volume.h"
#include "ImGuiFileDialog.h"
#include "color_mapper.h"
#include "histogram.h"
#include "imgui.h"
#include "imgui_stdlib.h"
#include "implot.h"
#include "log.h"
#include "slicer.h"
#include "volproc.h"
#include <GLFW/glfw3.h>

class gui {
public:
  gui();
  void InitWindow(int argc, char** argv);

private:
  const char* m_windowTitle = "Volume Processing Toolbox";
  ImVec4 m_clearColor = ImVec4(0.45f, 0.55f, 0.60f, 0.10f); // bg color

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

  void SetupWorkspace(ImGuiID& dockspace_id);

  bool showRaw = true;
  bool flagGpu = true; // should we process on the GPU

  // helper function to display stuff
  void ImImagesc(const float* data,
                 const uint64_t sizex,
                 const uint64_t sizey,
                 GLuint* out_texture,
                 const color_mapper myCMap);

  slicer mySlice; // used for preview of processed and unprocessed dataset

  // raw viz
  GLuint sliceZ;
  GLuint sliceX;
  GLuint sliceY;

  // dockspace ids
  ImGuiID m_dockLogs;
  ImGuiID m_dockToolsAnalyze;
  ImGuiID m_dockTools;

  // colomaps
  color_mapper rawMap;  // for raw data vizualization
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
};

#endif