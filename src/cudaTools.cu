#include "cudaTools.cuh"

void cudaTools::checkCudaErr(cudaError_t err, const char* msgErr)
{
	if (err != cudaSuccess)
	{
		printf("There was some CUDA error: %s, %s\n",
			msgErr, cudaGetErrorString(err));
		throw "CudaError";
	}
	return;
}