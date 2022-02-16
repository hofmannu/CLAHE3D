#include "gui.h"

gui::gui()
{
	niiReader.read_header("/home/hofmannu/Documents/MRT_brain.nii");
	niiReader.print_header();
	niiReader.read_data();
	histoEq.set_data(niiReader.get_pdataMatrix());
	histoEq.set_volSize(niiReader.get_dim());
	mySlice.set_sizeArray(niiReader.get_dim());
	mySlice.set_dataMatrix(niiReader.get_pdataMatrix());

	isDataLoaded = true;
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
	if (ImGui::Button("Load data"))
	{
		ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", 
			"Choose File", ".h5\0.mat\0.nii\0", ".");
	}

	if (ImGuiFileDialog::Instance()->FileDialog("ChooseFileDlgKey")) 
	{
		if (ImGuiFileDialog::Instance()->IsOk == true)
		{
			std::string inputFilePath = ImGuiFileDialog::Instance()->GetFilepathName();
			printf("File path: %s\n", inputFilePath.c_str());
			niiReader.read_header(inputFilePath);
			niiReader.print_header();
			niiReader.read_data();
			histoEq.set_data(niiReader.get_pdataMatrix());
			histoEq.set_volSize(niiReader.get_dim());
			mySlice.set_sizeArray(niiReader.get_dim());
			mySlice.set_dataMatrix(niiReader.get_pdataMatrix());

			isDataLoaded = true;
		}
		ImGuiFileDialog::Instance()->CloseDialog("ChooseFileDlgKey");
	}

	if (isDataLoaded)
	{
		ImGui::Text("Current dataset: %s", niiReader.get_filePath());
		ImGui::Text("Dimensions: %d x %d x %d", 
			niiReader.get_dim(0), niiReader.get_dim(1), niiReader.get_dim(2));
		ImGui::Text("Minimum value: %f", niiReader.get_min());
		ImGui::Text("Maximum value: %f", niiReader.get_max());
	}

	ImGui::End();
}

void gui::SettingsWindow()
{
	ImGui::Begin("Settings");
	ImGui::Checkbox("Overwrite flag", histoEq.get_pflagOverwrite());
	ImGui::Checkbox("GPU processing", &flagGpu);
	ImGui::InputInt3("Subvol spacing", histoEq.get_pspacingSubVols());
	ImGui::InputInt3("Subvol size", histoEq.get_psizeSubVols());
	ImGui::InputFloat("Noise level", histoEq.get_pnoiseLevel());
	ImGui::InputInt("Bin size", histoEq.get_pnBins());

	if (isDataLoaded)
	{
		if (ImGui::Button("CLAHE it!"))
		{
			if (flagGpu)
			{
				histoEq.calculate_cdf_gpu();
				histoEq.equalize_gpu();
			}
			else
			{
				histoEq.calculate_cdf();
				histoEq.equalize();	
			}
		}
	}

	ImGui::End();
}

void gui::SlicerWindow()
{
	if (isDataLoaded)
	{
		ImGui::Begin("Slicer");

		vector3<int> slicePos = mySlice.get_slicePoint();
		vector3<int> sizeArray = mySlice.get_sizeArray();
		ImGui::SliderInt("x", &slicePos.x, 0, sizeArray.x - 1);
		ImGui::SliderInt("y", &slicePos.y, 0, sizeArray.y - 1);
		ImGui::SliderInt("z", &slicePos.z, 0, sizeArray.z - 1);
		mySlice.set_slicePoint(slicePos);


		ImImagesc(
			mySlice.get_plane(0), 
			sizeArray.y, sizeArray.z, &rawSliceZ, rawMap);
		ImGui::Image((void*)(intptr_t)rawSliceZ, 
				ImVec2(550, 550),
				ImVec2(0, 0), // lower corner to crop
				ImVec2(sizeArray.y, sizeArray.z)); // upper corner to crop

		ImGui::SliderFloat("MinVal", 
			rawMap.get_pminVal(), 
			niiReader.get_min(), niiReader.get_max(), "%.1f");
		ImGui::SliderFloat("MaxVal", 
			rawMap.get_pmaxVal(), niiReader.get_min(), 
			niiReader.get_max(), "%.1f");
			

		ImGui::ColorEdit4("Min color", rawMap.get_pminCol(), ImGuiColorEditFlags_Float);
		ImGui::ColorEdit4("Max color", rawMap.get_pmaxCol(), ImGuiColorEditFlags_Float);
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