#include "volproc.h"

// class constructor and destructor
volproc::volproc()
{

}

volproc::~volproc()
{

}

// runs a mean filter over our output volume
void volproc::run_meanfilt(const meanfiltsett sett)
{
	new_entry("Running mean filter on volume...");
	meanfilter.set_dataInput(outputVol.get_pdata());
	meanfilter.set_dataSize(
		{outputVol.get_dim(0), outputVol.get_dim(1), outputVol.get_dim(2)});
	meanfilter.set_kernelSize(
		{sett.kernelSize[0], sett.kernelSize[1], sett.kernelSize[2]});
	meanfilter.run();
	new_entry("Finished mean filter execution");
	memcpy(outputVol.get_pdata(), meanfilter.get_pdataOutput(), 
		sizeof(float) * outputVol.get_nElements());

	new_entry("Recalculating maximum and minimum of output...");
	outputVol.calcMinMax();
	new_entry(" - minVal: " + std::to_string(outputVol.get_minVal()));
	new_entry(" - maxVal: " + std::to_string(outputVol.get_maxVal()));
	return;
}

// runs a gaussian filter over our output volume
void volproc::run_gaussfilt(const gaussfiltsett sett)
{
	new_entry("Running gauss filter on volume...");
	gaussfilter.set_dataInput(outputVol.get_pdata());
	gaussfilter.set_dataSize(
		{outputVol.get_dim(0), outputVol.get_dim(1), outputVol.get_dim(2)});
	gaussfilter.set_kernelSize(
		{sett.kernelSize[0], sett.kernelSize[1], sett.kernelSize[2]});
	gaussfilter.set_sigma(sett.sigma);
	gaussfilter.run();
	new_entry("Finished gauss filter execution");
	memcpy(outputVol.get_pdata(), gaussfilter.get_pdataOutput(), 
		sizeof(float) * outputVol.get_nElements());

	new_entry("Recalculating maximum and minimum of output...");
	outputVol.calcMinMax();
	new_entry(" - minVal: " + std::to_string(outputVol.get_minVal()));
	new_entry(" - maxVal: " + std::to_string(outputVol.get_maxVal()));
		
	return;
}

void volproc::run_thresholder(const thresholdersett sett)
{
	new_entry("Running thresholder on volume");
	thresfilter.set_minVal(sett.minVal);
	thresfilter.set_maxVal(sett.maxVal);
	thresfilter.threshold(outputVol.get_pdata(), outputVol.get_nElements());

	new_entry("Recalculating maximum and minimum of output...");
	outputVol.calcMinMax();
	new_entry(" - minVal: " + std::to_string(outputVol.get_minVal()));
	new_entry(" - maxVal: " + std::to_string(outputVol.get_maxVal()));
	return;
}

void volproc::run_histeq(const histeqsett sett)
{
	new_entry("Running histogram equilization...");
	histeqfilter.set_nBins(sett.nBins);
	histeqfilter.set_noiseLevel(sett.noiseLevel);
	histeqfilter.set_data(outputVol.get_pdata());
	histeqfilter.set_overwrite(1);
	histeqfilter.set_sizeSubVols({sett.sizeSubVols[0], sett.sizeSubVols[1], sett.sizeSubVols[2]});
	histeqfilter.set_spacingSubVols({sett.spacingSubVols[0], sett.spacingSubVols[1], sett.spacingSubVols[2]});
	histeqfilter.set_volSize({outputVol.get_dim(0), outputVol.get_dim(1), outputVol.get_dim(2)});
	histeqfilter.calculate_cdf();
	histeqfilter.equalize();
	new_entry("Done with histogram equilization...");
	
	new_entry("Recalculating maximum and minimum of output...");
	outputVol.calcMinMax();
	new_entry(" - minVal: " + std::to_string(outputVol.get_minVal()));
	new_entry(" - maxVal: " + std::to_string(outputVol.get_maxVal()));


	return;
}

void volproc::run_normalizer(const normalizersett<float> sett)
{
	new_entry("Running normalization on volume");
	new_entry(" - minVal: " + std::to_string(sett.minVal));
	new_entry(" - maxVal: " + std::to_string(sett.maxVal));
	normfilter.set_minVal(sett.minVal);
	normfilter.set_maxVal(sett.maxVal);
	normfilter.normalize(outputVol.get_pdata(), outputVol.get_nElements());

	new_entry("Recalculating maximum and minimum of output...");
	outputVol.calcMinMax();
	new_entry(" - minVal: " + std::to_string(outputVol.get_minVal()));
	new_entry(" - maxVal: " + std::to_string(outputVol.get_maxVal()));
	return;
}

// load dataset from file
void volproc::reload()
{
	new_entry("Loading dataset from " + inputPath);
	inputVol.readFromFile(inputPath);
	status += "Calculating minimum and maximum of datset\n";
	inputVol.calcMinMax();
	new_entry("Calculating histogram of input dataset");
	inputHist.calculate(inputVol.get_pdata(), inputVol.get_nElements());
	isDataLoaded = 1;

	outputVol = inputVol;
	return;
}

// resets the processed volume to the initial input volume
void volproc::reset()
{
	new_entry("Resetting volume to original values...");
	outputVol = inputVol;
	return;
}

// load a dataset from a defined path
void volproc::load(const string _filePath)
{
	inputPath = _filePath;
	reload();
	return;
}