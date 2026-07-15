/*
	unit test for the histogram class
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
*/

#include <catch2/catch_test_macros.hpp>

#include "../src/histogram.h"
#include <vector>

TEST_CASE("histogram counts every element for a normal dataset", "[histogram]")
{
	const std::size_t nBins = 10;
	histogram hist(nBins);

	float data[8] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
	hist.calculate(data, 8);

	// every element must be counted exactly once
	const std::vector<float> counts = hist.get_counter();
	float total = 0;
	for (float c : counts)
		total += c;
	REQUIRE(total == 8.0f);

	REQUIRE(hist.get_minVal() == 0.0f);
	REQUIRE(hist.get_maxVal() == 7.0f);
}

TEST_CASE("histogram handles a constant dataset without NaN/out-of-bounds", "[histogram]")
{
	// regression: for a constant dataset maxVal == minVal, so binSize == 0 and
	// currBin = (value - min) / 0 was NaN. Casting NaN to std::size_t is UB and
	// the `currBin >= nBins` guard does not catch NaN, so counter[currBin]++ could
	// write out of bounds. Everything must instead land in a single valid bin.
	const std::size_t nBins = 10;
	histogram hist(nBins);

	float data[5] = {5.0f, 5.0f, 5.0f, 5.0f, 5.0f};
	hist.calculate(data, 5);

	const std::vector<float> counts = hist.get_counter();
	REQUIRE(counts.size() == nBins);

	float total = 0;
	for (float c : counts)
		total += c;
	REQUIRE(total == 5.0f); // all elements counted, none lost or written OOB
}
