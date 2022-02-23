/*
	File: clahe3d_scripted.cpp
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 18.02.2022

	A rather simple script which will allow using clahe3d from the command line

	Usage:

	clahe3d_scripted $sv_spacing_in $sv_size $clip_value $nBins $nifti_in $nifti_out
	
	Where 
		sv_spacing: distance between subvolumes (uniform)
		sv_size: size of subvolumes
		clip_value: noise level which will be applied
		nBins: number of bins during histogram making
		nifti_in: input file in nii format
		nifi_out: output path for processed dataset

*/

#include <fstream>
#include <iostream>
#include "histeq.h"
#include "vector3.h"
#include "nifti.h"

using namespace std;

int main(int argc, char** argv)
{
	// lets try to read all the properties passed from the command line
	if (argc != 7)
	{
		printf("This function requires exactly 5 input arguments");
		printf("You have entered %d arguments\n", argc);
	  for (int i = 0; i < argc; ++i)
        cout << argv[i] << "\n";
		return 1;
	}

	const int sv_spacing = atoi(argv[1]);
	const int sv_size = atoi(argv[2]);
	const float noiseLevel = atof(argv[3]);
	const int nBins = atoi(argv[4]);
	const string pathIn = argv[5];
	const string pathOut = argv[6];

	printf("Input arguments\n");
	printf("- subvolume spacing: %d\n", sv_spacing);
	printf("- subvolume size: %d\n", sv_size);
	printf("- noiseLevel: %f\n", noiseLevel);
	printf("- nBins: %d\n", nBins);
	printf("- input file: %s\n", pathIn.c_str());
	printf("- output file: %s\n", pathOut.c_str());

	// load input file
	printf("Loading data...\n");
	nifti niiInput;
	niiInput.read(pathIn); // reads the entire input file
	// niiHandler.print_header();

	// prepare histogram handler
	histeq histHandler;
	histHandler.set_nBins(nBins);
	histHandler.set_noiseLevel(noiseLevel);
	histHandler.set_sizeSubVols({sv_size, sv_size, sv_size});
	histHandler.set_spacingSubVols({sv_spacing, sv_spacing, sv_spacing});
	histHandler.set_data(niiInput.get_pdataMatrix());
	histHandler.set_volSize(niiInput.get_dim());

	printf("Running CLAHE3D...\n");
	histHandler.calculate_cdf();
	histHandler.equalize();

	nifti niiOutput;
	niiOutput.set_header(niiInput.get_header());
	niiOutput.set_dataMatrix(histHandler.get_ptrOutput());

	printf("Saving data...\n");
	niiOutput.save(pathOut);

	return 0;
}