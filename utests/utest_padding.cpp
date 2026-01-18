/*
	File: utest_padding.cpp
	Author: Urs Hofmann
	Mail: maiL@hofmannu.org
	Date: 31.03.2022

	Description: checks if the padding of a volume through genfilt class works
*/

#include <catch2/catch.hpp>
#include "../src/genfilt.h"

TEST_CASE("volume padding through genfilt", "[genfilt][padding]")
{
	const std::size_t nKernel = 5;
	const std::size_t nx = 100;
	const std::size_t ny = 110;
	const std::size_t nz = 120;

	const std::size_t nkoff = (nKernel - 1) / 2;

	const std::size_t nxPadded = nx + nKernel - 1;
	const std::size_t nyPadded = ny + nKernel - 1;
	// const std::size_t nzPadded = nz + nKernel - 1;

	genfilt myFilt;
	myFilt.set_kernelSize({nKernel, nKernel, nKernel});
	myFilt.set_dataSize({nx, ny, nz});

	float* inputData = new float[myFilt.get_nData()];
	for (std::size_t iElem = 0; iElem < myFilt.get_nData(); iElem++)
	{
		inputData[iElem] = ((float) rand()) / ((float) RAND_MAX);
	}
	myFilt.set_dataInput(inputData);

	myFilt.padd();

	// check if everything went correct
	float* dataPadded = myFilt.get_pdataPadded();
	for (std::size_t iz = 0; iz < nz; iz++)
	{
		for (std::size_t iy = 0; iy < ny; iy++)
		{
			for (std::size_t ix = 0; ix < nx; ix++)
			{
				const std::size_t idxIn = ix + nx * (iy + ny * iz);
				const std::size_t idxOut = (ix + nkoff) + nxPadded * (iy + nkoff + nyPadded * (iz + nkoff));
				REQUIRE(inputData[idxIn] == dataPadded[idxOut]);
			}
		}
	}
}