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
#include "vector3.h"

using namespace std;

class gridder
{
private:
public:
	// variables
	vector3<int64_t> volSize; // size of full three dimensional volume [z, x, y]
	vector3<int64_t> nSubVols; // number of subvolumes in zxy
	vector3<int64_t> sizeSubVols; // size of each subvolume in zxy (should be uneven)
	vector3<int64_t> spacingSubVols; // spacing of subvolumes (they can overlap)
	vector3<int64_t> origin; // position of the very first element [0, 0, 0]
	vector3<int64_t> endIdx; // terminal value
	int64_t nElements; // total number of elements
	
	// calculations
	void calculate_nsubvols();

	// set functions
	void set_volSize(const vector3<int64_t>& _volSize); // overall size of the volume [z, x, y]
	void set_sizeSubVols(const vector3<int64_t>& _subVolSize); // size of each subvolume [z, x, y]
	void set_spacingSubVols(const vector3<int64_t>& _spacingSubVols); // space between subvolumes
	
	// get functions
	int64_t get_nSubVols(const uint8_t iDim) const {return nSubVols[iDim];};
	int64_t get_nSubVols() const {return nSubVols.x * nSubVols.y * nSubVols.z;};
	void get_neighbours(const vector3<int64_t>& position, int64_t* neighbours, float* ratio) const;
	vector3<int64_t> get_startIdxSubVol(const vector3<int64_t> iSub) const;
	vector3<int64_t> get_stopIdxSubVol(const vector3<int64_t> iSub) const;
	int64_t get_nElements() const;

};

#endif