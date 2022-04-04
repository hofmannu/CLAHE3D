#ifndef MEDIANFILT_ARGUMENTS
#define MEDIANFILT_ARGUMENTS

struct medianfilt_args {
	unsigned int volSize[3]; // size of base volume
	unsigned int kernelSize[3]; // size of kernel we invesigate
	unsigned int volSizePadded[3]; // size of padded volume
	unsigned int nKernel; // simply the number of elements which our kernel has
	float* inputData; // matrix holding all elements
};

#endif

#ifndef MEDIANFILT_KERNEL
#define MEDIANFILT_KERNEL 

template<typename T>
__device__ __inline__ void swap(T* const num1, T* const num2)
{
	const T backup = *num1;
	*num1 = *num2;
	*num2 = backup;
	return;
} 

// makes sure that the last element of this array is at the correct position
template<typename dataType, typename indexType>
__device__ unsigned int partition(dataType* arr, indexType left, indexType right)
{
  const dataType pivot = arr[right];
  indexType i = left;
  for (indexType j = left; j <= (right - 1); j++)
  {
      if (arr[j] <= pivot) 
      {
        swap(&arr[i], &arr[j]);
        i++;
      }
  }
  swap(&arr[i], &arr[right]);
  return i;
}
 
// This function returns k'th smallest element in arr[l..r] using QuickSort
// based method.  ASSUMPTION: ALL ELEMENTS IN ARR[] ARE DISTINCT
// k is the position of the element we want
template<typename dataType, typename indexType>
__device__ const dataType kth_smallest(dataType* arr, indexType left, indexType right, indexType k)
{
  // If k is smaller than number of elements in array
  if ((k > 0) && (k <= (right - left + 1))) 
  {
    // Partition the array around last element and get position of pivot element in sorted array
    const indexType index = partition(arr, left, right);

    // If position is same as k
    if ((index - left) == (k - 1))
      return arr[index];

    // If position is more, recur for left subarray
    if ((index - left) > (k - 1))
      return kth_smallest(arr, left, index - 1, k);
    else // Else recur for right subarray
    	return kth_smallest(arr, index + 1, right, k - index + left - 1);
  }

  // If k is more than number of elements in array
  return INT_MAX;
}

__global__ void medianfilt_cuda(float* outputData, const medianfilt_args args)
{
	const unsigned int idxVol[3] = {
		threadIdx.x + blockIdx.x * blockDim.x,
		threadIdx.y + blockIdx.y * blockDim.y,
		threadIdx.z + blockIdx.z * blockDim.z
	};

	// linearize thread index for shared memory
	const unsigned int idxThread = threadIdx.x + blockDim.x * 
		(threadIdx.y + blockDim.y * threadIdx.z);

	// check if we are operating in the boundaries of our volume
	if (
		(idxVol[0] < args.volSize[0]) &&
		(idxVol[1] < args.volSize[1]) &&
		(idxVol[2] < args.volSize[2]))
	{
		// unsigned int offset = idxVol[0] + 
		// float* localVec = &tempMatrix[];
		extern __shared__ float totalVec[];
		unsigned int threadOffset = idxThread * args.nKernel;
		float* localVec = &totalVec[threadOffset];

		// load elements to local sorting vector
		unsigned int localIdx = 0;
		unsigned int xAbs, yOffset, zOffset;
		for (unsigned int iz = 0; iz < args.kernelSize[2]; iz++)
		{
			zOffset = (iz + idxVol[2]) * args.volSizePadded[0] * args.volSizePadded[1];
			for (unsigned int iy = 0; iy < args.kernelSize[1]; iy++)
			{
				yOffset = (iy + idxVol[1]) * args.volSizePadded[0];
				for (unsigned int ix = 0; ix < args.kernelSize[0]; ix++)
				{
					// localV
					xAbs = ix + idxVol[0];
					localIdx = ix + args.kernelSize[0] * (iy + args.kernelSize[1] * iz);
					localVec[localIdx] = args.inputData[xAbs + yOffset + zOffset];
				}
			}
		}

		// extract nth element at center of kernel length
		const float medianVal = kth_smallest(localVec, 0u, args.nKernel - 1, (args.nKernel - 1) / 2 + 1);
		
		// write output
		const unsigned int idxOut = idxVol[0] + args.volSize[0] * 
			(idxVol[1] + args.volSize[1] * idxVol[2]);
		outputData[idxOut] = medianVal;
	}

	return;
}

#endif
