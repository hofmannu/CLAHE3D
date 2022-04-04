
// a small structure containing all the things we pass as constant arguments
#ifndef GENFILT_ARGS_H
#define GENFILT_ARGS_H

struct genfilt_args
{
	unsigned int volSize[3];
	unsigned int volSizePadded[3];
	
	unsigned int kernelSize[3];
	unsigned int nKernel;
	
	unsigned int localSize[3];
	unsigned int nLocal;

	float* inputData;
	float* kernel;
};


#endif

#ifndef GENFILT_CUDA_H
#define GENFILT_CUDA_H

__global__ void genfilt_cuda(float* outputData, const genfilt_args args)
{
	// index of voxel in outputData matrix
	const unsigned int idxVol[3] = {
		threadIdx.x + blockIdx.x * blockDim.x,
		threadIdx.y + blockIdx.y * blockDim.y,
		threadIdx.z + blockIdx.z * blockDim.z
	};

	const unsigned int idxThread = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
	const unsigned int nThreads = blockDim.x * blockDim.y * blockDim.z;

	// index in padded volume for 0, 0, 0 thread
	const unsigned int blockOffset[3] = {
		blockIdx.x * blockDim.x,
		blockIdx.y * blockDim.y,
		blockIdx.z * blockDim.z
	};

	// get pointers to shared memory
	extern __shared__ float ptrShared[];
	float* kernelLocal = &ptrShared[0];
	float* dataLocal = &ptrShared[args.nKernel];

	// load entire kernel into shared memory
	for (unsigned int sOffset = 0; sOffset < args.nKernel; sOffset += nThreads)
	{
		const unsigned int currId = sOffset + idxThread;

		if (currId < args.nKernel)
		{
			kernelLocal[currId] = args.kernel[currId];
		}
	}
	// __syncthreads();

	float tempVal = 0;
	const unsigned int nPlane = args.localSize[0] * args.localSize[1];

	for (unsigned int iz = 0; iz < args.localSize[2]; iz++)
	{
		// load next plane into memory
		for (unsigned int sOffset = 0; sOffset < nPlane; sOffset += nThreads)
		{
			const unsigned int currId = sOffset + idxThread;
			if (currId < nPlane)
			{
				const unsigned int iy = currId / args.localSize[0];
				const unsigned int ix = currId - iy * args.localSize[0];
				const unsigned int ixAbs = ix + blockOffset[0];
				const unsigned int iyAbs = iy + blockOffset[1];
				const unsigned int izAbs = iz + blockOffset[2];

				if ((ixAbs < args.volSizePadded[0]) &&
					(iyAbs < args.volSizePadded[1]) &&
					(izAbs < args.volSizePadded[2]))
				{
					dataLocal[currId] = args.inputData[ixAbs + args.volSizePadded[0] * 
						(iyAbs + izAbs * args.volSizePadded[1])];
				}
				else
				{
					dataLocal[currId] = 0;
				}

			}
		}
		__syncthreads();

		// multiply plane and kernel
		if ((iz >= threadIdx.z) && (iz < (threadIdx.z + args.kernelSize[2]) ))
		{
			const unsigned int zKernel = iz - threadIdx.z ;
			for (unsigned int iy = 0; iy < args.kernelSize[1]; iy++)
			{
				const unsigned int yAbs = threadIdx.y + iy;
				for (unsigned int ix = 0; ix < args.kernelSize[0]; ix++)
				{
					const unsigned int xAbs = threadIdx.x + ix;

					const unsigned int idxKernel = ix + args.kernelSize[0] * 
						(iy + args.kernelSize[1] * zKernel);
					const unsigned int idxPlane = xAbs + args.localSize[0] * yAbs;
								
					tempVal = fmaf(dataLocal[idxPlane], kernelLocal[idxKernel], tempVal);
					// if ((idxVol[0] == 0) && (idxVol[1] == 0) && (idxVol[2] == 0))
					// {
					// 	printf("(%u, %u, %u) \n", ix, iy, zKernel);
					// 	printf("treadIdx.z = %u, iz = %u\n", threadIdx.z, iz);
					// }
				}
			}
		}
		__syncthreads();
	}
	
	
	if (
		(idxVol[0] < args.volSize[0]) && 
		(idxVol[1] < args.volSize[1]) && 
		(idxVol[2] < args.volSize[2]))
	{
		const unsigned int idxOut = idxVol[0] + args.volSize[0] * (idxVol[1] + args.volSize[1] * idxVol[2]);
		outputData[idxOut] = tempVal;
	}

	return;
}

#endif