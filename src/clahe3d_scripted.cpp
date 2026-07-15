/*
	File: clahe3d_scripted.cpp
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 18.02.2022

	Command line frontend for the CLAHE3D algorithm. Run with --help to see all
	available options and their defaults.
*/

#include <cstdlib>
#include <iostream>
#include <string>

#include <cxxopts.hpp>

#include "histeq.h"
#include "vector3.h"
#include "../lib/CVolume/src/volume.h"

int main(int argc, char** argv)
{
	cxxopts::Options options(
		"clahe3d_scripted",
		"Contrast limited adaptive histogram equalization (CLAHE) for 3D volumes.");

	// Defaults mirror the histeqsett struct in histeq.h.
	options.add_options()
		("s,spacing", "Distance between subvolume centers [voxel]",
			cxxopts::value<std::size_t>()->default_value("5"))
		("z,size", "Edge length of each cubic subvolume [voxel]",
			cxxopts::value<std::size_t>()->default_value("11"))
		("n,noise", "Lower clip intensity of the histogram (noise level)",
			cxxopts::value<float>()->default_value("0.1"))
		("b,bins", "Number of histogram bins",
			cxxopts::value<std::size_t>()->default_value("255"))
		("i,input", "Input volume file (.nii / .h5)",
			cxxopts::value<std::string>())
		("o,output", "Output volume file",
			cxxopts::value<std::string>())
#if USE_CUDA
		("g,gpu", "Run the equalization on the GPU (CUDA)",
			cxxopts::value<bool>()->default_value("false"))
#endif
		("h,help", "Print usage");

	// Allow `clahe3d_scripted <input> <output>` in addition to -i/-o.
	options.parse_positional({"input", "output"});
	options.positional_help("<input> <output>");

	std::size_t sv_spacing, sv_size, nBins;
	float noiseLevel;
	std::string pathIn, pathOut;
	bool useGpu = false;

	try
	{
		const auto result = options.parse(argc, argv);

		if (result.count("help"))
		{
			std::cout << options.help() << std::endl;
			return 0;
		}

		if (!result.count("input") || !result.count("output"))
		{
			std::cerr << "Error: both an input and an output file are required.\n\n"
			          << options.help() << std::endl;
			return 1;
		}

		sv_spacing = result["spacing"].as<std::size_t>();
		sv_size = result["size"].as<std::size_t>();
		noiseLevel = result["noise"].as<float>();
		nBins = result["bins"].as<std::size_t>();
		pathIn = result["input"].as<std::string>();
		pathOut = result["output"].as<std::string>();
#if USE_CUDA
		useGpu = result["gpu"].as<bool>();
#endif
	}
	catch (const cxxopts::exceptions::exception& e)
	{
		std::cerr << "Error parsing options: " << e.what() << "\n\n"
		          << options.help() << std::endl;
		return 1;
	}

	printf("Input arguments\n");
	printf("- subvolume spacing [voxel]: %zu\n", sv_spacing);
	printf("- subvolume size [voxel]: %zu\n", sv_size);
	printf("- noiseLevel: %f\n", noiseLevel);
	printf("- nBins: %zu\n", nBins);
	printf("- input file: %s\n", pathIn.c_str());
	printf("- output file: %s\n", pathOut.c_str());
	printf("- backend: %s\n", useGpu ? "GPU" : "CPU");

	// load input file
	printf("Loading data...\n");
	volume niiInput;
	niiInput.readFromFile(pathIn); // reads the entire input file

	// prepare histogram handler
	histeq histHandler;
	histHandler.set_nBins(nBins);
	histHandler.set_noiseLevel(noiseLevel);
	histHandler.set_sizeSubVols({sv_size, sv_size, sv_size});
	histHandler.set_spacingSubVols({sv_spacing, sv_spacing, sv_spacing});
	histHandler.set_data(niiInput.get_pdata());
	histHandler.set_volSize({niiInput.get_dim(0), niiInput.get_dim(1), niiInput.get_dim(2)});
	histHandler.set_overwrite(1);

	printf("Running CLAHE3D...\n");
	if (useGpu)
	{
#if USE_CUDA
		histHandler.calculate_cdf_gpu();
		histHandler.equalize_gpu();
#endif
	}
	else
	{
		histHandler.calculate_cdf();
		histHandler.equalize();
	}

	printf("Saving data...\n");
	niiInput.saveToFile(pathOut);

	return 0;
}
