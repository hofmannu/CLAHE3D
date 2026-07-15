/*
	File: histogram_scripted.cpp
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 06.02.2022

	Command line frontend for the histogram analysis. Run with --help to see all
	available options and their defaults.
*/

#include <cstdlib>
#include <iostream>
#include <string>

#include <cxxopts.hpp>

#include "histogram.h"
#include "../lib/CVolume/src/volume.h"

int main(int argc, char** argv)
{
	cxxopts::Options options(
		"histogram_scripted",
		"Compute the intensity histogram of a 3D volume and write it to a file.");

	options.add_options()
		("b,bins", "Number of histogram bins",
			cxxopts::value<std::size_t>()->default_value("256"))
		("i,input", "Input volume file (.nii / .h5) to analyse",
			cxxopts::value<std::string>())
		("o,output", "Output file for the histogram",
			cxxopts::value<std::string>())
		("h,help", "Print usage");

	// Allow `histogram_scripted <input> <output>` in addition to -i/-o.
	options.parse_positional({"input", "output"});
	options.positional_help("<input> <output>");

	std::size_t nbins;
	std::string pathIn, pathOut;

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

		nbins = result["bins"].as<std::size_t>();
		pathIn = result["input"].as<std::string>();
		pathOut = result["output"].as<std::string>();
	}
	catch (const cxxopts::exceptions::exception& e)
	{
		std::cerr << "Error parsing options: " << e.what() << "\n\n"
		          << options.help() << std::endl;
		return 1;
	}

	printf("Input arguments\n");
	printf("- number of bins: %zu\n", nbins);
	printf("- input file: %s\n", pathIn.c_str());
	printf("- output file: %s\n", pathOut.c_str());

	volume niiInput;
	niiInput.readFromFile(pathIn); // reads the entire input file

	histogram hist(nbins);
	hist.calculate(niiInput.get_pdata(), niiInput.get_nElements());

	hist.print_to_file(pathOut);

	return 0;
}
