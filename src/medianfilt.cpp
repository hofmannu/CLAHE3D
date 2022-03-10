#include "medianfilt.h"

/* Function to sort an array using insertion sort*/
template<typename T>
void insertion_sort(T* arr, int n)
{
  int i;
  T key;
  int j;
  for (i = 1; i < n; i++)
  {
    key = arr[i];
    j = i - 1;

    /* Move elements of arr[0..i-1], that are
    greater than key, to one position ahead
    of their current position */
    while (j >= 0 && arr[j] > key)
    {
      arr[j + 1] = arr[j];
      j = j - 1;
    }
    arr[j + 1] = key;
  }
}
 

void medianfilt::run()
{
	padd();
	alloc_output();
	const auto tStart = std::chrono::high_resolution_clock::now();
	
	// temprary array used for sorting
	float* sortArray = new float [kernelSize.x * kernelSize.y * kernelSize.z];
	
	auto nKernel = kernelSize.x * kernelSize.y * kernelSize.z;

	for (auto iz = 0; iz < dataSize.z; iz++)
	{
		for (auto iy = 0; iy < dataSize.y; iy++)
		{
			for (auto ix = 0; ix < dataSize.x; ix++)
			{
				// for each output element we do the sum with the local kernel
				// lets read all subvalues into temporary array
				for (int zrel = 0; zrel < kernelSize.z; zrel++)
				{
					for (int yrel = 0; yrel < kernelSize.y; yrel++)
					{
						for (int xrel = 0; xrel < kernelSize.x; xrel++)
						{
							// get current index in padded volume
							const int xAbs = ix + xrel;
							const int yAbs = iy + yrel;
							const int zAbs = iz + zrel;

							// index in padded volume
							const int idxPadd = xAbs + 
								paddedSize.x * (yAbs + paddedSize.y * zAbs);

							const int idxKernel = xrel + 
								kernelSize.x * (yrel + kernelSize.y * zrel);

							sortArray[idxKernel] = dataPadded[idxPadd];
						}
					}
				}

				insertion_sort<float>(sortArray, nKernel);

				const int idxOut = ix + dataSize.x * (iy + dataSize.y * iz);
				dataOutput[idxOut] = sortArray[(nKernel - 1) / 2];
			}
		}
	}
	const auto tStop = std::chrono::high_resolution_clock::now();
	const auto tDuration = std::chrono::duration_cast<std::chrono::milliseconds>(tStop- tStart);
	tExec = tDuration.count();

	delete[] sortArray;
	return;
}