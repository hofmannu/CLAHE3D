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
	vector3<int> volSize; // size of full three dimensional volume [z, x, y]
	vector3<int> nSubVols; // number of subvolumes in zxy
	vector3<int> sizeSubVols = {11, 11, 11}; // size of each subvolume in zxy (should be uneven)
	vector3<int> spacingSubVols = {5, 5, 5}; // spacing of subvolumes (they can overlap)
	vector3<int> origin; // position of the very first element [0, 0, 0]
	vector3<int> endIdx; // terminal value
	int nElements; // total number of elements
	
	// calculations
	void calculate_nsubvols();

	// set functions
	void set_volSize(const vector3<int>& _volSize); // overall size of the volume [z, x, y]
	void set_sizeSubVols(const vector3<int>& _subVolSize); // size of each subvolume [z, x, y]
	void set_spacingSubVols(const vector3<int>& _spacingSubVols); // space between subvolumes
	
	int* get_pspacingSubVols() { return &spacingSubVols.x;};
	int* get_psizeSubVols() { return &sizeSubVols.x;};

	// get functions
	int get_nSubVols(const uint8_t iDim) const {return nSubVols[iDim];};
	int get_nSubVols() const {return nSubVols.x * nSubVols.y * nSubVols.z;};
	void get_neighbours(const vector3<int>& position, int* neighbours, float* ratio) const;
	vector3<int> get_startIdxSubVol(const vector3<int> iSub) const;
	vector3<int> get_stopIdxSubVol(const vector3<int> iSub) const;
	int get_nElements() const;

};

#endif