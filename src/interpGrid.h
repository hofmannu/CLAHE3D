#include <fstream>
#include<cstdint>

// defines a regularly spaced grid of points

class interpGrid{

	private:
		uint64_t gridSize[3]; // size of subvolume grid
		uint64_t volumeSize[3]; // dimensions of input volume in [z, x, y]
		uint64_t gridSpacing[3]; // spacing of grid in [z, x, y]
		float gridOrigin[3]; // position of first grid points in [z, x, y]
		float gridEnd[3]; // position of last grid point
		float remainder[3]; // size of last element

	public:
		
		void getNeighbours(
			uint64_t* position, // position of voxel to evaluate in [z, x, y]
			uint64_t* neighbours, // returned idx in grid [z0, z1, x0, x1, y0, y1]
			float* ratio); // weight along the three dimension
			// ratio order: ratioZ, ratioX, ratioY

		void calcSubVols();

		// set functions to define our grid
		void setGridSpacing(uint64_t* _gridSpacing);
		void setGridOrigin(float* _gridOrigin);
		void setVolumeSize(uint64_t* _volumeSize);

};
