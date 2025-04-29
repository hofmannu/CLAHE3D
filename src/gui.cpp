#include "gui.h"

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_internal.h"

// small helper functions to read size_t values
namespace ImGui {
/// \brief reads a std::size_t in by routing it throught the InputInt field
void InputInt(const char* message, std::size_t* value) {
  int tempVal = *value;
  ImGui::InputInt(message, &tempVal);
  *value = tempVal;
  return;
}

void InputInt3(const char* message, std::size_t* value) {
  int tempVal[3] = {(int)value[0], (int)value[1], (int)value[2]};
  ImGui::InputInt3(message, &tempVal[0]);
  value[0] = tempVal[0];
  value[1] = tempVal[1];
  value[2] = tempVal[2];
}
} // end of namespace ImGui

gui::gui() {
  inputVol = proc.get_pinputVol();
  outputVol = proc.get_poutputVol();
  inputHist = proc.get_pinputHist();
  outputHist = proc.get_poutputHist();
}

// displays a small help marker next to the text
static void HelpMarker(const char* desc) {
  ImGui::TextDisabled("(?)");
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
    ImGui::TextUnformatted(desc);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
  }
  return;
}

static void glfw_error_callback(int error, const char* description) {
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

void gui::SetupWorkspace(ImGuiID& dockspace_id) {
  ImGui::DockBuilderRemoveNode(dockspace_id);
  ImGuiDockNodeFlags dockSpaceFlags = 0;
  ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
  ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->Size);

  m_dockTools =
      ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Right, 0.20f, NULL, &dockspace_id);
  m_dockToolsAnalyze =
      ImGui::DockBuilderSplitNode(m_dockTools, ImGuiDir_Up, 0.40f, NULL, &m_dockTools);
  m_dockLogs = ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Down, 0.20f, NULL, &dockspace_id);

  ImGui::DockBuilderDockWindow("Data loader", m_dockTools);
  ImGui::DockBuilderDockWindow("Processing", m_dockToolsAnalyze);
  ImGui::DockBuilderDockWindow("Logger", m_dockLogs);
  ImGui::DockBuilderDockWindow("Slicer", dockspace_id);
  ImGui::DockBuilderDockWindow("Data exporter", m_dockTools);
}

void gui::InitWindow(int argcp, char** argv) {
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit()) {
    throw std::runtime_error("Failed to initialize GLFW");
  }

#if defined(IMGUI_IMPL_OPENGL_ES2)
  // GL ES 2.0 + GLSL 100
  const char* glsl_version = "#version 100";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
  // GL 3.2 + GLSL 150
  const char* glsl_version = "#version 150";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // 3.2+ only
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);           // Required on Mac
#else
  // GL 3.0 + GLSL 130
  const char* glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
#endif

  // todo add window flags again here
  GLFWwindow* window = glfwCreateWindow(1024 * 2, 512 * 2, m_windowTitle, nullptr, nullptr);
  if (window == nullptr) {
    throw std::runtime_error("Failed to create GLFW window");
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1); // Enable vsync

  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.ConfigWindowsMoveFromTitleBarOnly = true;

  // Setup Platform/Renderer backend
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  bool firstUse = true;

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGuiID dockspaceId = ImGui::DockSpaceOverViewport(ImGui::GetMainViewport(),
                                                       ImGuiDockNodeFlags_PassthruCentralNode);

    // if we are here for the first time, let's setup the workspace
    if (firstUse) {
      firstUse = false;
      SetupWorkspace(dockspaceId);
      ImGui::DockBuilderFinish(dockspaceId);
    }
    MainDisplayCode();
    ImGui::Render();

    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);

    glClearColor(m_clearColor.x * m_clearColor.w,
                 m_clearColor.y * m_clearColor.w,
                 m_clearColor.z * m_clearColor.w,
                 m_clearColor.w);

    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    // Update and Render additional Platform Windows
    // (Platform functions may change the current OpenGL context, so we
    // save/restore it to make it easier to paste this code elsewhere.
    //  For this specific demo app we could also call
    //  glfwMakeContextCurrent(window) directly)
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
      GLFWwindow* backup_current_context = glfwGetCurrentContext();
      ImGui::UpdatePlatformWindows();
      ImGui::RenderPlatformWindowsDefault();
      glfwMakeContextCurrent(backup_current_context);
    }

    glfwSwapBuffers(window);
  }
  // Cleanup
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();
}

// main loop running the display code
void gui::MainDisplayCode() {
  DataLoaderWindow();
  SettingsWindow();
  SlicerWindow();
  Console();
  ExportData();
}

void gui::Console() {
  const vector<logentry> myLog = proc.get_log();
  ImGui::Begin("Logger");
  ImGui::Columns(4);
  if (ImGui::Button("Clear")) {
    proc.clear_log();
  }
  ImGui::SameLine();
  HelpMarker("Clear console");
  // TODO add option to save console to file
  ImGui::NextColumn();
  ImGui::Checkbox("Log", proc.get_pflagLog());
  ImGui::NextColumn();
  ImGui::Checkbox("Warnings", proc.get_pflagWarning());
  ImGui::NextColumn();
  ImGui::Checkbox("Errors", proc.get_pflagError());
  ImGui::NextColumn();

  ImGui::Columns(1);
  ImGui::BeginChild("Scrolling");
  for (uint64_t iElem = 0; iElem < myLog.size(); iElem++) {
    // make date slightly faded out
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(125, 125, 125, 255));
    ImGui::Text(myLog[iElem].tString.c_str());
    ImGui::PopStyleColor();

    // TODO: ones we have errors and warnings make sure to adapt the font color
    // here
    ImGui::SameLine();
    ImGui::Text(myLog[iElem].message.c_str());
  }

  ImGui::EndChild();
  ImGui::End();
  return;
}

// small helper function which checks the limits agains maximum and minimum of
// dataset and adapts
void gui::check_mapLimits() {
  if (rawMap.get_maxVal() > inputVol->get_maxVal()) rawMap.set_maxVal(inputVol->get_maxVal());

  if (rawMap.get_minVal() < inputVol->get_minVal()) rawMap.set_minVal(inputVol->get_minVal());

  if (procMap.get_maxVal() > outputVol->get_maxVal()) procMap.set_maxVal(outputVol->get_maxVal());

  if (procMap.get_minVal() < outputVol->get_minVal()) procMap.set_minVal(outputVol->get_minVal());

  return;
}

// dialog to export our dataset to a file
void gui::ExportData() {
  if (proc.get_isDataLoaded()) {
    ImGui::Begin("Data exporter");
    ImGui::InputText("Output folder", &outputPath);
    ImGui::InputText("Output file name", &outputFile);

    if (ImGui::Button("Export")) {
      // if outputPath does not end on backlash, append it
      if (outputPath.back() != '/') outputPath += '/';

      outputVol->saveToFile(outputPath + outputFile);
    }
    ImGui::End();
  }
  return;
}

// window to load datasets from the harddrive
void gui::DataLoaderWindow() {
  ImGui::Begin("Data loader");
  ImGui::Columns(3);

  if (ImGui::Button("Load data")) {
    ImGuiFileDialog::Instance()->OpenDialog(
        "ChooseFileDlgKey", "Choose File", ".nii,.hdf,.raw,.h5", "h5", ".");
  }

  // display
  if (ImGuiFileDialog::Instance()->Display(
          "ChooseFileDlgKey", ImGuiWindowFlags_NoCollapse, {500.0f, 400.0f})) {
    // action if OK
    if (ImGuiFileDialog::Instance()->IsOk()) {
      proc.load(ImGuiFileDialog::Instance()->GetFilePathName());

      // also set all the color bar limits now to the range of the dataset
      rawMap.set_minVal(inputVol->get_minVal());
      rawMap.set_maxVal(inputVol->get_maxVal());
      procMap.set_minVal(inputVol->get_minVal());
      procMap.set_maxVal(inputVol->get_maxVal());

      mySlice.set_sizeArray({inputVol->get_dim(0), inputVol->get_dim(1), inputVol->get_dim(2)});
      mySlice.set_dataMatrix(inputVol->get_pdata());
    }
    ImGuiFileDialog::Instance()->Close();
  }

  // if we loaded data once already we can allow for reloads
  if (!proc.get_isDataLoaded()) {
    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
  }

  ImGui::NextColumn();
  if (ImGui::Button("Reset")) {
    proc.reset();
  }
  if (!proc.get_isDataLoaded()) {
    ImGui::PopItemFlag();
    ImGui::PopStyleVar();
  }

  ImGui::SameLine();
  HelpMarker("This will reset the dataset to the initial status without any "
             "filtering applied.");

  ImGui::NextColumn();

  if (!proc.get_isDataLoaded()) {
    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
  }

  if (ImGui::Button("Reload")) {
    proc.reload();
  }

  if (!proc.get_isDataLoaded()) {
    ImGui::PopItemFlag();
    ImGui::PopStyleVar();
  }

  ImGui::SameLine();
  HelpMarker("This will reload the file from the disc.");

  ImGui::Columns(1);
  if (proc.get_isDataLoaded()) {
    if (ImGui::CollapsingHeader("Dataset information")) {
      ImGui::Columns(2);
      ImGui::Text("File path");
      ImGui::NextColumn();
      ImGui::Text("%s", proc.get_inputPath());
      ImGui::NextColumn();
      ImGui::Text("Dimensions");
      ImGui::NextColumn();
      ImGui::Text(
          "%lu x %lu x %lu", inputVol->get_dim(0), inputVol->get_dim(1), inputVol->get_dim(2));
      ImGui::NextColumn();
      ImGui::Text("Resolution");
      ImGui::NextColumn();
      ImGui::Text(
          "%.2f x %.3f %.3f", inputVol->get_res(0), inputVol->get_res(1), inputVol->get_res(2));
      ImGui::NextColumn();
      ImGui::Text("Data range");
      ImGui::NextColumn();
      ImGui::Text("%.3f ... %.3f", inputVol->get_minVal(), inputVol->get_maxVal());
      ImGui::Columns(1);
    }

    if (ImGui::CollapsingHeader("Raw data histogram")) {
      // plot histogram of input data
      ImPlot::BeginPlot("Histogram input volume");
      // ImPlot::SetupAxes
      // conf.values.xs = inputHist->get_pcontainerVal(); // this line is
      // optional std::vector<float> yVec = inputHist->get_counter();
      // conf.values.ys = &yVec[0]; // this line is optional
      // conf.values.count = inputHist->get_nBins();
      // conf.scale.min = inputHist->get_minHist();
      // conf.scale.max = inputHist->get_maxHist();
      // conf.tooltip.show = true;
      // conf.tooltip.format = "x=%.2f, y=%.2f";
      // conf.grid_x.show = true;
      // conf.grid_y.show = true;
      // conf.frame_size = ImVec2(400, 200);
      // conf.line_thickness = 2.f;
      // ImGui::Plot(, conf);
      ImPlot::EndPlot();
    }
  }

  ImGui::End();
}

void gui::SettingsWindow() {
  if (proc.get_isDataLoaded()) {
    ImGui::Begin("Processing");
    // ImGui::Checkbox("Overwrite flag", histoEq.get_pflagOverwrite());

    if (ImGui::CollapsingHeader("CLAHE3D")) {

      ImGui::InputInt3("Subvol spacing", sett_histeq.spacingSubVols);
      ImGui::InputInt3("Subvol size", sett_histeq.sizeSubVols);
      ImGui::InputFloat("Noise level", &sett_histeq.noiseLevel);
      ImGui::InputInt("Bin size", &sett_histeq.nBins);

      if (ImGui::Button("CLAHE it!")) {
        proc.run_histeq(sett_histeq);
        check_mapLimits();
      }
    }

    // applying mean filter
    if (ImGui::CollapsingHeader("Mean filter")) {
      ImGui::InputInt3("Kernel size", sett_meanfilt.kernelSize);
      ImGui::SameLine();
      HelpMarker("Number of neighbouring voxels taking into account during "
                 "mean filter along x, y, z.");
#if USE_CUDA
      ImGui::Checkbox("GPU", &sett_meanfilt.flagGpu);
#endif
      if (ImGui::Button("Run mean filter")) {
        proc.run_meanfilt(sett_meanfilt);
        check_mapLimits();
      }
    }

    // applying a gaussian filter to our volume
    if (ImGui::CollapsingHeader("Gaussian filter")) {
      ImGui::InputInt3("Kernel size", sett_gaussfilt.kernelSize);
      ImGui::InputFloat("Sigma", &sett_gaussfilt.sigma);
#if USE_CUDA
      ImGui::Checkbox("GPU", &sett_gaussfilt.flagGpu);
#endif
      if (ImGui::Button("Run gaussian filter")) {
        proc.run_gaussfilt(sett_gaussfilt);
        check_mapLimits();
      }
    }

    if (ImGui::CollapsingHeader("Median filter")) {
      ImGui::InputInt3("Kernel size", sett_medianfilt.kernelSize);
#if USE_CUDA
      ImGui::Checkbox("GPU", &sett_medianfilt.flagGpu);
#endif
      ImGui::SameLine();
      HelpMarker("Number of neighbouring voxels taking into account during "
                 "median filter along x, y, z.");
      if (ImGui::Button("Run median filter")) {
        proc.run_medianfilt(sett_medianfilt);
        check_mapLimits();
      }
    }

    // thresholding of volume against some values
    if (ImGui::CollapsingHeader("Thresholder")) {
      ImGui::InputFloat("Lower threshold", &sett_thresholdfilt.minVal);
      ImGui::InputFloat("Upper threshold", &sett_thresholdfilt.maxVal);
      if (ImGui::Button("Run thresholding")) {
        proc.run_thresholder(sett_thresholdfilt);
        check_mapLimits();
      }
    }

    // normalize the data range of the volume
    if (ImGui::CollapsingHeader("Normalizer")) {
      ImGui::InputFloat("Lower value", &sett_normalizer.minVal);
      ImGui::InputFloat("Upper value", &sett_normalizer.maxVal);
      if (ImGui::Button("Run normalizer")) {
        proc.run_normalizer(sett_normalizer);
        check_mapLimits();
      }
    }

    ImGui::End();
  }

  return;
}

void gui::SlicerWindow() {

  if (proc.get_isDataLoaded()) {
    ImGui::Begin("Slicer");

    ImGui::Columns(4);
    if (ImGui::Button("Flip x")) {
      mySlice.flip(0);
    }
    ImGui::NextColumn();

    if (ImGui::Button("Flip y")) {
      mySlice.flip(1);
    }
    ImGui::NextColumn();

    if (ImGui::Button("Flip z")) {
      mySlice.flip(2);
    }
    ImGui::NextColumn();

    bool oldRaw = showRaw;
    ImGui::Checkbox("Show raw", &showRaw);
    // if we switched toggle, lets update data pointer
    if (oldRaw != showRaw) {
      if (showRaw) mySlice.set_dataMatrix(inputVol->get_pdata());
      else
        mySlice.set_dataMatrix(outputVol->get_pdata());
    }

    ImGui::Columns(1);
    vector3<std::size_t> slicePos = mySlice.get_slicePoint();
    vector3<std::size_t> sizeArray = mySlice.get_sizeArray();
    ImGui::Columns(3);
    vector3<int> intedPos = {(int)slicePos[0], (int)slicePos[1], (int)slicePos[2]};
    ImGui::SliderInt("x", &intedPos.x, 0, sizeArray.x - 1);
    ImGui::NextColumn();
    ImGui::SliderInt("y", &intedPos.y, 0, sizeArray.y - 1);
    ImGui::NextColumn();
    ImGui::SliderInt("z", &intedPos.z, 0, sizeArray.z - 1);
    ImGui::Columns(1);
    mySlice.set_slicePoint(intedPos.x, intedPos.y, intedPos.z);

    ImGui::Text("Value at current position (raw): %f",
                (showRaw) ? inputVol->get_value(slicePos.x, slicePos.y, slicePos.z)
                          : outputVol->get_value(slicePos.x, slicePos.y, slicePos.z));

    const float totalHeight = inputVol->get_length(1) + inputVol->get_length(2);
    const float totalWidth = inputVol->get_length(0) + inputVol->get_length(1);
    const float maxSize = 1000;
    const float scale = maxSize / ((totalWidth > totalHeight) ? totalWidth : totalHeight);
    const int xLength = round(scale * inputVol->get_length(0));
    const int yLength = round(scale * inputVol->get_length(1));
    const int zLength = round(scale * inputVol->get_length(2));

    if (showRaw) {
      ImImagesc(mySlice.get_plane(0), sizeArray.y, sizeArray.z, &sliceX, rawMap);
      ImImagesc(mySlice.get_plane(1), sizeArray.x, sizeArray.z, &sliceY, rawMap);
      ImImagesc(mySlice.get_plane(2), sizeArray.x, sizeArray.y, &sliceZ, rawMap);

      ImGui::Image((void*)(intptr_t)sliceY, ImVec2(xLength, zLength));
      ImGui::SameLine();
      ImGui::Image((void*)(intptr_t)sliceX, ImVec2(yLength, zLength));
      ImGui::Image((void*)(intptr_t)sliceZ, ImVec2(xLength, yLength));

      ImGui::Columns(2);
      ImGui::SliderFloat("Min Val Raw",
                         rawMap.get_pminVal(),
                         inputVol->get_minVal(),
                         inputVol->get_maxVal(),
                         "%.4f");
      ImGui::NextColumn();
      ImGui::SliderFloat("Max Val Raw",
                         rawMap.get_pmaxVal(),
                         inputVol->get_minVal(),
                         inputVol->get_maxVal(),
                         "%.4f");

      ImGui::NextColumn();
      ImGui::ColorEdit4("Min color Raw", rawMap.get_pminCol(), ImGuiColorEditFlags_Float);
      ImGui::NextColumn();
      ImGui::ColorEdit4("Max color Raw", rawMap.get_pmaxCol(), ImGuiColorEditFlags_Float);
    } else {
      ImImagesc(mySlice.get_plane(0), sizeArray.y, sizeArray.z, &sliceX, procMap);
      ImImagesc(mySlice.get_plane(1), sizeArray.x, sizeArray.z, &sliceY, procMap);
      ImImagesc(mySlice.get_plane(2), sizeArray.x, sizeArray.y, &sliceZ, procMap);

      ImGui::Image((void*)(intptr_t)sliceY, ImVec2(xLength, zLength));
      ImGui::SameLine();
      ImGui::Image((void*)(intptr_t)sliceX, ImVec2(yLength, zLength));
      ImGui::Image((void*)(intptr_t)sliceZ, ImVec2(xLength, yLength));

      ImGui::Columns(2);
      ImGui::SliderFloat("Min Val Proc",
                         procMap.get_pminVal(),
                         outputVol->get_minVal(),
                         outputVol->get_maxVal(),
                         "%.4f");
      ImGui::NextColumn();
      ImGui::SliderFloat("Max Val Proc",
                         procMap.get_pmaxVal(),
                         outputVol->get_minVal(),
                         outputVol->get_maxVal(),
                         "%.4f");
      ImGui::NextColumn();
      ImGui::ColorEdit4("Min color Proc", procMap.get_pminCol(), ImGuiColorEditFlags_Float);
      ImGui::NextColumn();
      ImGui::ColorEdit4("Max color Proc", procMap.get_pmaxCol(), ImGuiColorEditFlags_Float);
    }
    ImGui::End();
  }
  return;
}

// helper function to display stuff
void gui::ImImagesc(const float* data,
                    const uint64_t sizex,
                    const uint64_t sizey,
                    GLuint* out_texture,
                    const color_mapper myCMap) {

  glDeleteTextures(1, out_texture);

  // Create an OpenGL texture identifier
  GLuint image_texture;
  glGenTextures(1, &image_texture);
  glBindTexture(GL_TEXTURE_2D, image_texture);

  // setup filtering parameters for display
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  // use color transfer function to convert from float to rgba
  std::vector<unsigned char> data_conv(4 * sizex * sizey);
  myCMap.convert_to_map(data, sizex * sizey, data_conv.data());
  glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
  glTexImage2D(
      GL_TEXTURE_2D, 0, GL_RGBA, sizex, sizey, 0, GL_RGBA, GL_UNSIGNED_BYTE, data_conv.data());

  // give pointer back to main program
  *out_texture = image_texture;
}