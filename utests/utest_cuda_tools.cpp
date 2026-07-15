#include <catch2/catch_test_macros.hpp>

#include "../src/cudaTools.cuh"

TEST_CASE("cudaTools can query device properties", "[cuda]")
{
	cudaTools myTools;
	REQUIRE_NOTHROW(myTools.print_devProps());
}
