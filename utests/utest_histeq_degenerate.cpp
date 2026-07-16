/*
	Regression tests for histeq on degenerate sub-volumes and for the storage layout
	of its index accessors.
*/

#include <catch2/catch_test_macros.hpp>

#include "../src/histeq.h"
#include "../src/vector3.h"
#include <vector>
#include <cmath>

namespace
{
	// histeq manages raw pointers and is not copyable, so configure it in place
	// rather than returning it by value.
	void configure(histeq& hist, const vector3<std::size_t>& volSize, float fillValue,
		float noiseLevel, std::vector<float>& storage)
	{
		storage.assign(volSize.elementMult(), fillValue);
		hist.set_nBins(16);
		hist.set_noiseLevel(noiseLevel);
		hist.set_volSize(volSize);
		hist.set_sizeSubVols({3, 3, 3});
		hist.set_spacingSubVols({2, 2, 2});
		hist.set_overwrite(0);
		hist.set_data(storage.data());
	}
}

TEST_CASE("histeq CDF is finite when all voxels are identical (single-bin sub-volumes)", "[histeq]")
{
	// regression: a constant sub-volume puts every above-noise voxel in one bin, so
	// the cdf span was 0 and 1/valRange -> inf produced NaN across the cdf.
	std::vector<float> data;
	histeq hist;
	configure(hist, {9, 9, 9}, 0.5f, 0.1f, data);

	hist.calculate_cdf();
	for (std::size_t i = 0; i < hist.get_ncdf(); i++)
		REQUIRE(std::isfinite(hist.get_cdf(i)));

#if USE_CUDA
	// the GPU path must agree and also stay finite
	std::vector<float> refCdf(hist.get_ncdf());
	for (std::size_t i = 0; i < hist.get_ncdf(); i++)
		refCdf[i] = hist.get_cdf(i);

	hist.calculate_cdf_gpu();
	for (std::size_t i = 0; i < hist.get_ncdf(); i++)
	{
		REQUIRE(std::isfinite(hist.get_cdf(i)));
		REQUIRE(hist.get_cdf(i) == refCdf[i]);
	}
#endif
}

TEST_CASE("histeq CDF fills the last bin for an all-below-noise sub-volume", "[histeq]")
{
	// regression: the no-signal fallback looped to nBins-1, leaving the last cdf bin
	// uninitialised (garbage). It must be a full linear ramp ending at 1.0.
	std::vector<float> data;
	histeq hist;
	configure(hist, {9, 9, 9}, 0.0f, 0.5f, data); // everything below noise level

	hist.calculate_cdf();

	const std::size_t nBins = 16;
	const std::size_t nSub = hist.get_nSubVols();
	for (std::size_t iSub = 0; iSub < nSub; iSub++)
	{
		for (std::size_t iBin = 0; iBin < nBins; iBin++)
			REQUIRE(std::isfinite(hist.get_cdf(iBin, iSub)));
		REQUIRE(hist.get_cdf(0, iSub) == 0.0f);
		REQUIRE(hist.get_cdf(nBins - 1, iSub) == 1.0f);
	}
}
