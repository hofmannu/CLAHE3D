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
	vector3<std::size_t> volSize; // size of full three dimensional volume [z, x, y]
	vector3<std::size_t> nSubVols; // number of subvolumes in zxy
	vector3<std::size_t> sizeSubVols = {11, 11, 11}; // size of each subvolume in zxy (should be uneven)
	vector3<std::size_t> spacingSubVols = {5, 5, 5}; // spacing of subvolumes (they can overlap)
	vector3<std::size_t> origin; // position of the very first element [0, 0, 0]
	vector3<std::size_t> endIdx; // terminal value
	std::size_t nElements; // total number of elements
	
	// calculations
	void calculate_nsubvols();

	// set functions
	void set_volSize(const vector3<std::size_t>& _volSize); // overall size of the volume [z, x, y]
	void set_sizeSubVols(const vector3<std::size_t>& _subVolSize); // size of each subvolume [z, x, y]
	void set_spacingSubVols(const vector3<std::size_t>& _spacingSubVols); // space between subvolumes
	
	std::size_t* get_pspacingSubVols() { return &spacingSubVols.x;};
	std::size_t* get_psizeSubVols() { return &sizeSubVols.x;};

	// get functions
	std::size_t get_nSubVols(const uint8_t iDim) const {return nSubVols[iDim];};
	std::size_t get_nSubVols() const {return nSubVols.x * nSubVols.y * nSubVols.z;};
	void get_neighbours(const vector3<std::size_t>& position, std::size_t* neighbours, float* ratio) const;
	
	vector3<std::size_t> get_startIdxSubVol(const vector3<std::size_t>& iSub) const;
		vector3<std::size_t> get_stopIdxSubVol(const vector3<std::size_t>& iSub) const;
	
	std::size_t get_nElements() const;

};

#endif