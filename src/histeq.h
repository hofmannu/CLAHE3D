#include <fstream>
#include <cstdint>

class histeq{

	private:
		// private variables
		uint64_t nBins; // number of histogram bins
		float noiseLevel; // noise level threshold (clipLimit)
		uint64_t volSize[3]; // size of full three dimensional volume
		float* dataMatrix; // 3d matrix containing all values
		uint64_t nSubVols[3]; // number of subvolumes in zxy
		uint64_t sizeSubVols[3]; // size of each subvolume in zxy
		float* cdf; // contains cummulative distribution function for each subol
		// structure: iBin + iZ * nBins + iX * nBins * nZ + iY * nBins * nZ * nX 
		float* icdf; // inverted version of cummulative distribution function
		float* maxVal; // maximum value occuring in each bin [iZ, iX, iY]
		float overallMax = 0;

		// private functions
		void calculate_nsubvols();
		void getCDF(
			const uint64_t zStart, const uint64_t zEnd,
			const uint64_t xStart, const uint64_t xEnd,
			const uint64_t yStart, const uint64_t yEnd, 
			const uint64_t iZ, const uint64_t iX, const uint64_t iY);
		void invertCDF(uint64_t iZ, uint64_t iX, uint64_t iY);
		void getOverallMax();

	public:
		void calculate();
		float get_icdf(
			const uint64_t iZ, const uint64_t iX, const uint64_t iY, const float value);

		void setNBins(const uint64_t _nBins);
		void setNoiseLevel(const float _noiseLevel);
		void setVolSize(const uint64_t* _volSize);
		void setData(float* _dataMatrix);
		void setSizeSubVols(const uint64_t* _subVolSize);

};
