
/*
	A master class which can do all the processing without anyone
	
	File: volproc.h
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 08.03.2022
*/

#ifndef VOLPROC_H
#define VOLPROC_H

#include "../lib/CVolume/src/volume.h"
#include "histogram.h"
#include "meanfilt.h"
#include "gaussfilt.h"
#include "medianfilt.h"
#include "thresholder.h"
#include "histeq.h"
#include "log.h"
#include "normalizer.h"

class volproc : public log
{
public:
	// class constructor and destructor
	volproc();
	~volproc();

	void reload();
	void reset();
	void load(const string _inputPath);

	[[nodiscard]] bool get_isDataLoaded() const {return isDataLoaded;};
	[[nodiscard]] const char* get_inputPath() const {return inputPath.c_str();};

	// pointers to all volumes and datasets
	[[nodiscard]] volume* get_pinputVol() {return &inputVol;};
	[[nodiscard]] volume* get_poutputVol() {return &outputVol;};
	[[nodiscard]] const histogram* get_pinputHist() const {return &inputHist;};
	[[nodiscard]] const histogram* get_poutputHist() const {return &outputHist;};

	[[nodiscard]] const char* get_status() const {return status.c_str();};

	// all processing functions go here
	void run_meanfilt(const meanfiltsett sett);
	void run_gaussfilt(const gaussfiltsett sett);
	void run_thresholder(const thresholdersett sett);
	void run_histeq(const histeqsett sett);
	void run_normalizer(const normalizersett<float> sett);
	void run_medianfilt(const medianfiltsett sett);

private:
	volume inputVol; //!< the input volume as stored before any processing
	volume outputVol; //!< the output volume after applying the processing

	histogram inputHist;
	histogram outputHist;

	std::string inputPath;
	bool isDataLoaded = false;

	string status;

	// different volume processing tools
	meanfilt meanfilter;
	gaussfilt gaussfilter;
	medianfilt medianfilter;
	thresholder thresfilter;
	histeq histeqfilter;
	normalizer<float> normfilter;


};

#endif