#include <catch2/catch.hpp>
#include "../src/cudaTools.cuh"

TEST_CASE("CUDA tools device properties", "[cuda][tools]")
{
	cudaTools myTools;
	myTools.print_devProps();
}