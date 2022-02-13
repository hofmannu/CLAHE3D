#include "cudaTools.cuh"

void cudaTools::checkCudaErr(cudaError_t err, const char* msgErr)
{
	if (err != cudaSuccess)
	{
		printf("There was some CUDA error appearing along my way: %s\n",
			cudaGetErrorString(err));
		throw "CudaError";
	}
	return;
}