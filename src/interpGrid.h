#include <fstream>
#include<cstdint>

// defines a regularly spaced grid of points

class interpGrid{

	private:
		uint64_t gridSize[3]; // dimensions of grid in [z, x, y]
		uint64_t gridSpacing[3]; // spacing of grid in [z, x, y]
		float gridOrigin[3]; // offset of first grid points in [z, x, y]
	
	public:
		
		void getNeighbours(
			uint64_t* position, // position of voxel to evaluate in [z, x, y]
			uint64_t* neighbours, // returned idx in grid [z0, z1, x0, x1, y0, y1]
			float* ratio); // weight along the three dimension
			// ratio order: ratioZ, ratioX, ratioY

		void getNBins();

		// set functions to define our grid
		void setGridSpacing(uint64_t* _gridSpacing);
		void setGridOrigin(float* _gridOrigin);
		void setGridSize(uint64_t* _gridSize);

};
