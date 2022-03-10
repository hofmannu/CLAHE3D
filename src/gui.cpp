#include "gui.h"

gui::gui()
{
	inputVol = proc.get_pinputVol();
	outputVol = proc.get_poutputVol();
	inputHist = proc.get_pinputHist();
	outputHist = proc.get_poutputHist();
}

gui::~gui()
{

}

// displays a small help marker next to the text
static void HelpMarker(const char* desc)
{
	ImGui::TextDisabled("(?)");
	if (ImGui::IsItemHovered())
	{
		ImGui::BeginTooltip();
		ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
		ImGui::TextUnformatted(desc);
		ImGui::PopTextWrapPos();
		ImGui::EndTooltip();
	}
	return;
}

void gui::InitWindow(int *argcp, char**argv)
{
	if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0)
	{
	  printf("Error: %s\n", SDL_GetError());
		return;
	}
	// main_display_function goes somewhere here
	const char* glsl_version = "#version 140";
	// to find out which glsl version you are using, run glxinfo from terminal
	// and look for "OpenGL shading language version string"
	// https://en.wikipedia.org/wiki/OpenGL_Shading_Language

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
	
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
	SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

	SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
	SDL_Window* window = SDL_CreateWindow(windowTitle, 
		SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1900, 1080, window_flags);
	SDL_GLContext gl_context = SDL_GL_CreateContext(window);
	SDL_GL_MakeCurrent(window, gl_context);
	SDL_GL_SetSwapInterval(1); // Enable vsync

	bool err = glewInit() != GLEW_OK;
	if (err)
	{
		printf("Failed to initialize OpenGL loader!");
	  throw "FailedOpenGLLoader";
	}

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;

	ImGui::StyleColorsDark();
	ImGui_ImplSDL2_InitForOpenGL(window, gl_context);

	ImGui_ImplOpenGL3_Init(glsl_version);
	bool done = false;
	while (!done)
	{
		SDL_Event event;
		while (SDL_PollEvent(&event))
		{
			ImGui_ImplSDL2_ProcessEvent(&event);
			if (event.type == SDL_QUIT)
				done = true;
		}
		// Start the Dear ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplSDL2_NewFrame(window);
		ImGui::NewFrame();
		MainDisplayCode();
		// Rendering
		ImGui::Render();
		
		glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
		glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
		glClear(GL_COLOR_BUFFER_BIT);
		//glUseProgram(0); // You may want this if using this code in an OpenGL 3+ context where shaders may be bound
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		SDL_GL_SwapWindow(window);
	}

	// Cleanup
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplSDL2_Shutdown();
	ImGui::DestroyContext();

	SDL_GL_DeleteContext(gl_context);
 	SDL_DestroyWindow(window);
 	SDL_Quit();
	return;
}

// main loop running the display code
void gui::MainDisplayCode()
{
	DataLoaderWindow();
	SettingsWindow();
	SlicerWindow();
	Console();
	return;
}

void gui::Console()
{
	ImGui::Begin("Log");
	ImGui::Text(proc.get_log().c_str());
	ImGui::End();
	return;
}

// window to load datasets from the harddrive
void gui::DataLoaderWindow()
{
	ImGui::Begin("Data loader");
	ImGui::Columns(3);
	if (ImGui::Button("Load data"))
	{
		ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", 
			"Choose File", ".nii\0.h5\0", ".");
	}

	if (ImGuiFileDialog::Instance()->FileDialog("ChooseFileDlgKey")) 
	{
		if (ImGuiFileDialog::Instance()->IsOk == true)
		{
			proc.load(ImGuiFileDialog::Instance()->GetFilepathName());
			mySlice.set_sizeArray(
				{inputVol->get_dim(0), inputVol->get_dim(1), inputVol->get_dim(2)});
			mySlice.set_dataMatrix(inputVol->get_pdata());
		}
		ImGuiFileDialog::Instance()->CloseDialog("ChooseFileDlgKey");
	}

	// if we loaded data once already we can allow for reloads
	if (!proc.get_isDataLoaded())
	{
		ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
	}

	ImGui::NextColumn();
	if (ImGui::Button("Reset"))
	{
		proc.reset();
	}
	if (!proc.get_isDataLoaded())
	{
		ImGui::PopItemFlag();
  	ImGui::PopStyleVar();
	}

	ImGui::SameLine();
	HelpMarker("This will reset the dataset to the initial status without any filtering applied.");

	ImGui::NextColumn();

	if (!proc.get_isDataLoaded())
	{
		ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
	}

	if (ImGui::Button("Reload"))
	{
		proc.reload();
	}
	
	if (!proc.get_isDataLoaded())
	{
		ImGui::PopItemFlag();
  	ImGui::PopStyleVar();
	}
	
	ImGui::SameLine();
	HelpMarker("This will reload the file from the disc.");

	ImGui::Columns(1);
	if (proc.get_isDataLoaded())
	{
		if (ImGui::CollapsingHeader("Dataset information"))
		{
			ImGui::Columns(2);
			ImGui::Text("File path"); ImGui::NextColumn(); 
			ImGui::Text("%s", proc.get_inputPath()); ImGui::NextColumn();
			ImGui::Text("Dimensions"); ImGui::NextColumn(); 
			ImGui::Text("%lu x %lu x %lu", 
				inputVol->get_dim(0), inputVol->get_dim(1), inputVol->get_dim(2)); ImGui::NextColumn(); 
			ImGui::Text("Resolution"); ImGui::NextColumn(); ImGui::Text("%.2f x %.3f %.3f", 
				inputVol->get_res(0), inputVol->get_res(1), inputVol->get_res(2)); ImGui::NextColumn(); 
			ImGui::Text("Data range"); ImGui::NextColumn(); ImGui::Text("%.3f ... %.3f", 
				inputVol->get_minVal(), inputVol->get_maxVal());
			ImGui::Columns(1);
		}

		if (ImGui::CollapsingHeader("Raw data histogram"))
		{
			// plot histogram of input data 
			ImGui::PlotConfig conf;
			conf.values.xs = inputHist->get_pcontainerVal(); // this line is optional
			conf.values.ys = inputHist->get_pcounter(); // this line is optional
			conf.values.count = inputHist->get_nBins();
			conf.scale.min = inputHist->get_minHist();
			conf.scale.max = inputHist->get_maxHist();
			conf.tooltip.show = true;
			conf.tooltip.format = "x=%.2f, y=%.2f";
			conf.grid_x.show = false;
			conf.grid_y.show = false;
			conf.frame_size = ImVec2(400, 200);
			conf.line_thickness = 2.f;
			ImGui::Plot("Histogram input volume", conf);
		}

	}

	ImGui::End();
}

void gui::SettingsWindow()
{
	if (proc.get_isDataLoaded())
	{
		ImGui::Begin("Processing");
		// ImGui::Checkbox("Overwrite flag", histoEq.get_pflagOverwrite());

		if (ImGui::CollapsingHeader("CLAHE3D"))
		{
	
			ImGui::InputInt3("Subvol spacing", sett_histeq.spacingSubVols);
			ImGui::InputInt3("Subvol size", sett_histeq.sizeSubVols);
			ImGui::InputFloat("Noise level", &sett_histeq.noiseLevel);
			ImGui::InputInt("Bin size", &sett_histeq.nBins);

			if (ImGui::Button("CLAHE it!"))
			{
				proc.run_histeq(sett_histeq);
			}
		}

		// applying mean filter
		if (ImGui::CollapsingHeader("Mean filter"))
		{
			ImGui::InputInt3("Kernel size", sett_meanfilt.kernelSize);
			ImGui::SameLine();
			HelpMarker("Number of neighbouring voxels taking into account during mean filter along x, y, z.");
			if (ImGui::Button("Run mean filter"))
			{
				proc.run_meanfilt(sett_meanfilt);
			}
		}

		// applying a gaussian filter to our volume
		if (ImGui::CollapsingHeader("Gaussian filter"))
		{
			ImGui::InputInt3("Kernel size", sett_gaussfilt.kernelSize);
			ImGui::InputFloat("Sigma", &sett_gaussfilt.sigma);
			if (ImGui::Button("Run gaussian filter"))
			{
				proc.run_gaussfilt(sett_gaussfilt);
			}
		}

		// thresholding of volume against some values
		if (ImGui::CollapsingHeader("Thresholder"))
		{
			ImGui::InputFloat("Lower threshold", &sett_thresholdfilt.minVal);
			ImGui::InputFloat("Upper threshold", &sett_thresholdfilt.maxVal);
			if (ImGui::Button("Run thresholding"))
			{
				proc.run_thresholder(sett_thresholdfilt);
			}
		}

		// normalize the data range of the volume
		if (ImGui::CollapsingHeader("Normalizer"))
		{
			ImGui::InputFloat("Lower value", &sett_normalizer.minVal);
			ImGui::InputFloat("Upper value", &sett_normalizer.maxVal);
			if (ImGui::Button("Run normalizer"))
			{
				proc.run_normalizer(sett_normalizer);
			}
		}

		ImGui::End();
	}

	return;
}

void gui::SlicerWindow()
{

	if (proc.get_isDataLoaded())
	{
		ImGui::Begin("Slicer");
	
		ImGui::Columns(4);	
		if (ImGui::Button("Flip x"))
		{
		 mySlice.flip(0); 
		}
		ImGui::NextColumn();

		if (ImGui::Button("Flip y"))
		{
		 mySlice.flip(1); 
		}
		ImGui::NextColumn();

		if (ImGui::Button("Flip z"))
		{
		 mySlice.flip(2);
		}
		ImGui::NextColumn();
		
		
		bool oldRaw = showRaw;
		ImGui::Checkbox("Show raw", &showRaw);
		// if we switched toggle, lets update data pointer
		if (oldRaw != showRaw)
		{
			if (showRaw)
				mySlice.set_dataMatrix(inputVol->get_pdata());
			else
				mySlice.set_dataMatrix(outputVol->get_pdata());
		}

		ImGui::Columns(1);
		vector3<int> slicePos = mySlice.get_slicePoint();
		vector3<int> sizeArray = mySlice.get_sizeArray();
		ImGui::Columns(3);
		ImGui::SliderInt("x", &slicePos.x, 0, sizeArray.x - 1);
		ImGui::NextColumn();
		ImGui::SliderInt("y", &slicePos.y, 0, sizeArray.y - 1);
		ImGui::NextColumn();
		ImGui::SliderInt("z", &slicePos.z, 0, sizeArray.z - 1);
		ImGui::Columns(1);
		mySlice.set_slicePoint(slicePos);

		ImGui::Text("Value at current position (raw): %f", 
			(showRaw) ? 
			inputVol->get_value(slicePos.x, slicePos.y, slicePos.z) : 
			outputVol->get_value(slicePos.x, slicePos.y, slicePos.z));

		// ImGui::Text("Value at current position (proc): %f", histoEq.get_outputValue(slicePos));

		const float totalHeight = inputVol->get_length(1) + inputVol->get_length(2);
		const float totalWidth = inputVol->get_length(0) + inputVol->get_length(1);
		const float maxSize = 1000;
		const float scale = maxSize / ((totalWidth > totalHeight) ? totalWidth : totalHeight);
		const int xLength = round(scale * inputVol->get_length(0));
		const int yLength = round(scale * inputVol->get_length(1));
		const int zLength = round(scale * inputVol->get_length(2));

		if (showRaw)
		{
			ImImagesc(mySlice.get_plane(0), sizeArray.y, sizeArray.z, &sliceX, rawMap);
			ImImagesc(mySlice.get_plane(1), sizeArray.x, sizeArray.z, &sliceY, rawMap);
			ImImagesc(mySlice.get_plane(2), sizeArray.x, sizeArray.y, &sliceZ, rawMap);

			ImGui::Image((void*)(intptr_t) sliceY, ImVec2(xLength, zLength)); ImGui::SameLine(); 
			ImGui::Image((void*)(intptr_t) sliceX, ImVec2(yLength, zLength));
			ImGui::Image((void*)(intptr_t) sliceZ, ImVec2(xLength, yLength)); 
			

			ImGui::Columns(2);
			ImGui::SliderFloat("Min Val Raw", 
				rawMap.get_pminVal(), inputVol->get_minVal(), inputVol->get_maxVal(), "%.4f");
			ImGui::NextColumn();
			ImGui::SliderFloat("Max Val Raw", 
				rawMap.get_pmaxVal(), inputVol->get_minVal(), inputVol->get_maxVal(), "%.4f");
				

			ImGui::NextColumn();
			ImGui::ColorEdit4("Min color Raw", rawMap.get_pminCol(), ImGuiColorEditFlags_Float);
			ImGui::NextColumn();
			ImGui::ColorEdit4("Max color Raw", rawMap.get_pmaxCol(), ImGuiColorEditFlags_Float);
		}
		else
		{
			ImImagesc(mySlice.get_plane(0),	sizeArray.y, sizeArray.z, &sliceX, procMap);
			ImImagesc(mySlice.get_plane(1),	sizeArray.x, sizeArray.z, &sliceY, procMap);
			ImImagesc(mySlice.get_plane(2),	sizeArray.x, sizeArray.y, &sliceZ, procMap);
			
			ImGui::Image((void*)(intptr_t) sliceY, ImVec2(xLength, zLength)); ImGui::SameLine(); 
			ImGui::Image((void*)(intptr_t) sliceX, ImVec2(yLength, zLength));
			ImGui::Image((void*)(intptr_t) sliceZ, ImVec2(xLength, yLength)); 
			
			ImGui::Columns(2);
			ImGui::SliderFloat("Min Val Proc", procMap.get_pminVal(), outputVol->get_minVal(), outputVol->get_maxVal(), "%.4f");
			ImGui::NextColumn();
			ImGui::SliderFloat("Max Val Proc", procMap.get_pmaxVal(), outputVol->get_minVal(), outputVol->get_maxVal(), "%.4f");
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
void gui::ImImagesc(
	const float* data, const uint64_t sizex, const uint64_t sizey, 
	GLuint* out_texture, const color_mapper myCMap)
{
	
	glDeleteTextures(1, out_texture);

	// Create an OpenGL texture identifier
	GLuint image_texture;
	glGenTextures(1, &image_texture);
	glBindTexture(GL_TEXTURE_2D, image_texture);

	// setup filtering parameters for display
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			
	// use color transfer function to convert from float to rgba
	unsigned char* data_conv = new unsigned char[4 * sizex * sizey];
	myCMap.convert_to_map(data, sizex * sizey, data_conv);
	glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, sizex, sizey, 0, GL_RGBA, GL_UNSIGNED_BYTE, data_conv);

	// give pointer back to main program
	*out_texture = image_texture;
	delete[] data_conv; // free memory for temporary array
	return;
}