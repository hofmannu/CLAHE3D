/*
	unit test for the thresholder class
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
*/

#include <catch2/catch_test_macros.hpp>

#include "../src/thresholder.h"
#include <vector>

TEST_CASE("thresholder clamps values into [minVal, maxVal]", "[thresholder]")
{
	thresholder th;
	th.set_minVal(0.2f);
	th.set_maxVal(0.8f);

	std::vector<float> data = {-1.0f, 0.0f, 0.5f, 0.9f, 2.0f};
	th.threshold(data.data(), data.size());

	REQUIRE(data[0] == 0.2f);
	REQUIRE(data[1] == 0.2f);
	REQUIRE(data[2] == 0.5f);
	REQUIRE(data[3] == 0.8f);
	REQUIRE(data[4] == 0.8f);
}

TEST_CASE("thresholder does not permanently swap its bounds when reversed", "[thresholder]")
{
	// regression: threshold() did `if (maxVal < minVal) std::swap(minVal, maxVal)`,
	// mutating the members, so a second call saw different bounds than were set.
	thresholder th;
	th.set_minVal(0.8f); // deliberately reversed
	th.set_maxVal(0.2f);

	std::vector<float> data = {0.5f};
	th.threshold(data.data(), data.size());

	// the stored bounds must be untouched
	REQUIRE(th.get_minVal() == 0.8f);
	REQUIRE(th.get_maxVal() == 0.2f);

	// and the clamp itself must use the ordered range [0.2, 0.8]
	REQUIRE(data[0] == 0.5f);
}
