/*
	File: histogram_scripted.cpp
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 06.02.2022

	A rather simple script which will allow using histogram from the command line

	Usage:

	histogram $nbins $nifti_in $nifti_out
	
	Where 
		nbins - number of bins to use for calculation e.g. 256
		nifit_in - dataset to analyse
		outfile - here we print result
*/

#include <fstream>
#include <iostream>
#include "histogram.h"
#include "../lib/CVolume/src/volume.h"

using namespace std;

int main(int argc, char** argv)
{
	// lets try to read all the properties passed from the command line
	if (argc != 4)
	{
		printf("This function requires exactly 3 input arguments");
		printf("You have entered %d arguments\n", argc);
	  for (int i = 0; i < argc; ++i)
        cout << argv[i] << "\n";
		return 1;
	}

	const std::size_t nbins = atoi(argv[1]);
	const string pathIn = argv[2];
	const string pathOut = argv[3];

	printf("Input arguments\n");
	printf("- number of bins: %d\n", nbins);
	printf("- input file: %s\n", pathIn.c_str());
	printf("- output file: %s\n", pathOut.c_str());

	volume niiInput;
	niiInput.readFromFile(pathIn); // reads the entire input file
	
	histogram hist(nbins);
	hist.calculate(niiInput.get_pdata(), niiInput.get_nElements());

	hist.print_to_file(pathOut);

	return 0;
}