/*
	Regression tests for the multi-threaded CPU paths when the number of worker
	threads exceeds the number of z-slices/sub-volumes.

	Bug: the range each worker processes was computed as `dataSize.z / nThreads`.
	When nThreads > dataSize.z this is 0, and the per-worker stop index
	`(iWorker + 1) * 0 - 1` underflows std::size_t to SIZE_MAX, so the worker loop
	writes far out of bounds (crash / heap corruption). nThreads defaults to
	std::thread::hardware_concurrency(), so any volume thinner than the core count
	was affected.

	Each test forces nThreads well above the z-extent and checks that the result
	is (a) produced without crashing and (b) identical to the single-threaded result.
*/

#include <catch2/catch_test_macros.hpp>

#include "../src/genfilt.h"
#include "../src/medianfilt.h"
#include "../src/histeq.h"
#include <vector>

namespace
{
	// deterministic, varied values in [0, 1)
	std::vector<float> makeInput(const std::size_t n)
	{
		std::vector<float> data(n);
		for (std::size_t i = 0; i < n; i++)
			data[i] = static_cast<float>((i * 37) % 100) / 100.0f;
		return data;
	}
}

TEST_CASE("genfilt::conv is correct when nThreads exceeds dataSize.z", "[genfilt][threading]")
{
	genfilt filt;
	filt.set_kernelSize({3, 3, 3});
	filt.set_dataSize({4, 4, 2}); // z = 2, smaller than the forced thread count

	std::vector<float> input = makeInput(filt.get_nData());
	std::vector<float> kernel(filt.get_nKernel(), 1.0f);
	filt.set_dataInput(input.data());
	filt.set_kernel(kernel.data());

	// single-threaded reference
	filt.set_nThreads(1);
	filt.conv();
	std::vector<float> ref(filt.get_pdataOutput(), filt.get_pdataOutput() + filt.get_nData());

	// many more threads than z-slices: used to underflow and write out of bounds
	filt.set_nThreads(16);
	filt.conv();
	const float* out = filt.get_pdataOutput();
	for (std::size_t i = 0; i < filt.get_nData(); i++)
		REQUIRE(out[i] == ref[i]);
}

TEST_CASE("medianfilt::run is correct when nThreads exceeds dataSize.z", "[medianfilt][threading]")
{
	medianfilt filt;
	filt.set_kernelSize({3, 3, 3});
	filt.set_dataSize({4, 4, 2});

	std::vector<float> input = makeInput(filt.get_nData());
	filt.set_dataInput(input.data());

	filt.set_nThreads(1);
	filt.run();
	std::vector<float> ref(filt.get_pdataOutput(), filt.get_pdataOutput() + filt.get_nData());

	filt.set_nThreads(16);
	filt.run();
	const float* out = filt.get_pdataOutput();
	for (std::size_t i = 0; i < filt.get_nData(); i++)
		REQUIRE(out[i] == ref[i]);
}

TEST_CASE("histeq::equalize is correct when nThreads exceeds volSize.z", "[histeq][threading]")
{
	const vector3<std::size_t> volSize(9, 9, 3); // z = 3, smaller than forced thread count

	histeq hist;
	hist.set_nBins(16);
	hist.set_noiseLevel(0.01f);
	hist.set_volSize(volSize);
	hist.set_sizeSubVols({3, 3, 3});
	hist.set_spacingSubVols({2, 2, 2});
	hist.set_overwrite(0);

	std::vector<float> input = makeInput(volSize.elementMult());
	hist.set_data(input.data());

	// cdf partition is already safe; compute it once
	hist.set_nThreads(1);
	hist.calculate_cdf();

	// single-threaded equalization reference
	hist.equalize();
	std::vector<float> ref(hist.get_nElements());
	for (std::size_t i = 0; i < hist.get_nElements(); i++)
		ref[i] = hist.get_outputValue(i);

	// many more threads than z-slices: used to underflow and write out of bounds
	hist.set_nThreads(16);
	hist.equalize();
	for (std::size_t i = 0; i < hist.get_nElements(); i++)
		REQUIRE(hist.get_outputValue(i) == ref[i]);
}
