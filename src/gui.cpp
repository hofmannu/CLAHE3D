#include "gui.h"

gui::gui()
{
	histoEq.set_overwrite(0);
	isDataLoaded = false;

	// prepare colormap for processed
	procMap.set_minVal(0.0);
	procMap.set_maxVal(1.0);
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


void gui::MainDisplayCode()
{
	DataLoaderWindow();
	SettingsWindow();
	SlicerWindow();
	return;
}

void gui::DataLoaderWindow()
{
	ImGui::Begin("Data loader");
	ImGui::Columns(2);
	if (ImGui::Button("Load data"))
	{
		ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", 
			"Choose File", ".nii\0", ".");
	}

	if (ImGuiFileDialog::Instance()->FileDialog("ChooseFileDlgKey")) 
	{
		if (ImGuiFileDialog::Instance()->IsOk == true)
		{
			std::string inputFilePath = ImGuiFileDialog::Instance()->GetFilepathName();
			printf("File path: %s\n", inputFilePath.c_str());
			niiReader.read(inputFilePath); // reads the entire input file
			niiReader.print_header();
			histoEq.set_data(niiReader.get_pdataMatrix());
			histoEq.set_volSize(niiReader.get_dim());
			histoEq.set_noiseLevel(niiReader.get_min());
			mySlice.set_sizeArray(niiReader.get_dim());
			mySlice.set_dataMatrix(niiReader.get_pdataMatrix());
			rawMap.set_maxVal(niiReader.get_max());
			rawMap.set_minVal(niiReader.get_min());
			histRawData.calculate(niiReader.get_pdataMatrix(), niiReader.get_nElements());
			showRaw = 1;
			isProc = 0;

			isDataLoaded = true;
		}
		ImGuiFileDialog::Instance()->CloseDialog("ChooseFileDlgKey");
	}

	// if we loaded data once already we can allow for reloads
	if (isDataLoaded)
	{
		ImGui::NextColumn();
		if (ImGui::Button("Reload"))
		{
			niiReader.read(); // reads the entire input file
		}
	}

	ImGui::Columns(1);
	if (isDataLoaded)
	{
		if (ImGui::CollapsingHeader("Dataset information"))
		{
			ImGui::Columns(2);
			ImGui::Text("File path"); ImGui::NextColumn(); ImGui::Text("%s", niiReader.get_filePath()); ImGui::NextColumn(); 
			ImGui::Text("Dimensions"); ImGui::NextColumn(); ImGui::Text("%d x %d x %d", 
				niiReader.get_dim(0), niiReader.get_dim(1), niiReader.get_dim(2)); ImGui::NextColumn(); 
			ImGui::Text("Resolution"); ImGui::NextColumn(); ImGui::Text("%.2f x %.3f %.3f", 
				niiReader.get_res(0), niiReader.get_res(1), niiReader.get_res(2)); ImGui::NextColumn(); 
			ImGui::Text("Data range"); ImGui::NextColumn(); ImGui::Text("%.3f ... %.3f", niiReader.get_min(), niiReader.get_max());
			ImGui::Columns(1);
		}

		if (ImGui::CollapsingHeader("Raw data histogram"))
		{
			// plot histogram of input data 
			ImGui::PlotConfig conf;
			conf.values.xs = histRawData.get_pcontainerVal(); // this line is optional
			conf.values.ys = histRawData.get_pcounter(); // this line is optional
			conf.values.count = histRawData.get_nBins();
			conf.scale.min = histRawData.get_minHist();
			conf.scale.max = histRawData.get_maxHist();
			conf.tooltip.show = true;
			conf.tooltip.format = "x=%.2f, y=%.2f";
			conf.grid_x.show = false;
			conf.grid_y.show = false;
			conf.frame_size = ImVec2(400, 200);
			conf.line_thickness = 2.f;
			ImGui::Plot("Histogram", conf);
		}

	}

	ImGui::End();
}

void gui::SettingsWindow()
{
	if (isDataLoaded)
	{
		ImGui::Begin("Settings");
		// ImGui::Checkbox("Overwrite flag", histoEq.get_pflagOverwrite());
		ImGui::Checkbox("GPU processing", &flagGpu);
		ImGui::InputInt3("Subvol spacing", histoEq.get_pspacingSubVols());
		ImGui::InputInt3("Subvol size", histoEq.get_psizeSubVols());
		ImGui::InputFloat("Noise level", histoEq.get_pnoiseLevel());
		ImGui::InputInt("Bin size", histoEq.get_pnBins());

		if (isDataLoaded)
		{
			if (ImGui::Button("CLAHE it!"))
			{
	#if USE_CUDA
				if (flagGpu)
				{
					histoEq.calculate_cdf_gpu();
					histoEq.equalize_gpu();
					isProc = 1; 
					showRaw = 0;
				}
				else
				{
	#endif
					histoEq.calculate_cdf();
					histoEq.equalize();
					isProc = 1;	
					showRaw = 0;
	#if USE_CUDA
				}
	#endif
			}
		}

		ImGui::End();
	}
}

void gui::SlicerWindow()
{

	if (isDataLoaded)
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
		if (isProc)
		{
			ImGui::Checkbox("Show raw", &showRaw);
		}

		// if we switched toggle, lets update data pointer
		if (oldRaw != showRaw)
		{
			if (showRaw)
				mySlice.set_dataMatrix(niiReader.get_pdataMatrix());
			else
				mySlice.set_dataMatrix(histoEq.get_ptrOutput());
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

		ImGui::Text("Value at current position (raw): %f", niiReader.get_val(slicePos));
		if (isProc)
			ImGui::Text("Value at current position (proc): %f", histoEq.get_outputValue(slicePos));

		int width = 500;
		int heightX = round(((float) width) / niiReader.get_length(1) * niiReader.get_length(0)); 
		int heightZ = round(((float) width) / niiReader.get_length(1) * niiReader.get_length(2)); 
		int widthZ = round(((float) heightX) / niiReader.get_length(0) * niiReader.get_length(2)); 

		if (showRaw)
		{
			ImImagesc(mySlice.get_plane(0), sizeArray.y, sizeArray.z, &sliceX, rawMap);
			ImImagesc(mySlice.get_plane(1), sizeArray.z, sizeArray.x, &sliceY, rawMap);
			ImImagesc(mySlice.get_plane(2), sizeArray.y, sizeArray.x, &sliceZ, rawMap);

			ImGui::Image((void*)(intptr_t) sliceZ, ImVec2(width, heightX)); ImGui::SameLine(); 
			ImGui::Image((void*)(intptr_t) sliceY, ImVec2(widthZ, heightX)); 
			ImGui::Image((void*)(intptr_t) sliceX, ImVec2(width, heightZ));
			

			ImGui::Columns(2);
			ImGui::SliderFloat("Min Val Raw", 
				rawMap.get_pminVal(), niiReader.get_min(), niiReader.get_max(), "%.1f");
			ImGui::NextColumn();
			ImGui::SliderFloat("Max Val Raw", 
				rawMap.get_pmaxVal(), niiReader.get_min(), niiReader.get_max(), "%.1f");
				

			ImGui::NextColumn();
			ImGui::ColorEdit4("Min color Raw", rawMap.get_pminCol(), ImGuiColorEditFlags_Float);
			ImGui::NextColumn();
			ImGui::ColorEdit4("Max color Raw", rawMap.get_pmaxCol(), ImGuiColorEditFlags_Float);
		}
		else
		{
			ImImagesc(mySlice.get_plane(0),	sizeArray.y, sizeArray.z, &sliceX, procMap);
			ImImagesc(mySlice.get_plane(1),	sizeArray.z, sizeArray.x, &sliceY, procMap);
			ImImagesc(mySlice.get_plane(2),	sizeArray.y, sizeArray.x, &sliceZ, procMap);
			ImGui::Image((void*)(intptr_t) sliceZ, ImVec2(width, heightX));  ImGui::SameLine(); 
			ImGui::Image((void*)(intptr_t) sliceY, ImVec2(widthZ, heightX)); 
			ImGui::Image((void*)(intptr_t) sliceX, ImVec2(width, widthZ)); 
			
			ImGui::Columns(2);
			ImGui::SliderFloat("Min Val Proc", procMap.get_pminVal(), 0, 1, "%.2f");
			ImGui::NextColumn();
			ImGui::SliderFloat("Max Val Proc", procMap.get_pmaxVal(), 0, 1, "%.2f");
			ImGui::NextColumn();
			ImGui::ColorEdit4("Min color Proc", procMap.get_pminCol(), ImGuiColorEditFlags_Float);
			ImGui::NextColumn();
			ImGui::ColorEdit4("Max color Proc", procMap.get_pmaxCol(), ImGuiColorEditFlags_Float);
		}
		ImGui::End();
	}
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