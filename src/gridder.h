/* 
	definition of  the grid used for our histogram calculations
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 13.02.2022
*/

#ifndef GRIDDER_H
#define GRIDDER_H

#include <cstdint>
#include <iostream>

using namespace std;

class gridder
{
private:
public:
	// variables
	uint64_t volSize[3]; // size of full three dimensional volume [z, x, y]
	uint64_t nSubVols[3]; // number of subvolumes in zxy
	uint64_t sizeSubVols[3]; // size of each subvolume in zxy (should be uneven)
	uint64_t spacingSubVols[3]; // spacing of subvolumes (they can overlap)
	uint64_t origin[3]; // position of the very first element [0, 0, 0]
	uint64_t end[3]; // terminal value
	uint64_t nElements; // total number of elements
	
	// calculations
	void calculate_nsubvols();

	// set functions
	void set_volSize(const uint64_t* _volSize); // overall size of the volume [z, x, y]
	void set_sizeSubVols(const uint64_t* _subVolSize); // size of each subvolume [z, x, y]
	void set_spacingSubVols(const uint64_t* _spacingSubVols); // space between subvolumes
	
	// get functions
	uint64_t get_nSubVols(const uint8_t iDim) const {return nSubVols[iDim];};
	uint64_t get_nSubVols() const {return nSubVols[0] * nSubVols[1] * nSubVols[2];};
	void get_neighbours(const uint64_t* position, uint64_t* neighbours, float* ratio) const;
	uint64_t get_startIdxSubVol(const uint64_t iSub, const uint8_t iDim) const;
	uint64_t get_stopIdxSubVol(const uint64_t iSub, const uint8_t iDim) const;
	uint64_t get_nElements() const;

};

#endif