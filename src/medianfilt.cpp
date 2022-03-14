#include "medianfilt.h"

// /* Function to sort an array using insertion sort*/
// template<typename T>
// inline void insertion_sort(T* arr, const int n)
// {
//   int i;
//   T key;
//   int j;
//   for (i = 1; i < n; i++)
//   {
//     key = arr[i];
//     j = i - 1;

//     /* Move elements of arr[0..i-1], that are
//     greater than key, to one position ahead
//     of their current position */
//     while (j >= 0 && arr[j] > key)
//     {
//       arr[j + 1] = arr[j];
//       j = j - 1;
//     }
//     arr[j + 1] = key;
//   }
//   return;
// }

// template<typename T>
// inline void swap(T* arr , int pos1, int pos2){
// 	T temp;
// 	temp = arr[pos1];
// 	arr[pos1] = arr[pos2];
// 	arr[pos2] = temp;
// 	return;
// }

// template<typename T>
// inline int partition(T* arr, int low, int high, T pivot){
// 	int i = low;
// 	int j = low;
// 	while( i <= high)
// 	{
// 		if(arr[i] > pivot)
// 		{
// 			i++;
// 		}
// 		else
// 		{
// 			swap(arr,i,j);
// 			i++;
// 			j++;
// 		}
// 	}
// 	return j-1;
// }

// template<typename T>
// inline void quick_sort(T* arr, int low, const int high){
// 	if(low < high)
// 	{
// 		T pivot = arr[high];
// 		int pos = partition<T>(arr, low, high, pivot);
		
// 		quick_sort(arr, low, pos-1);
// 		quick_sort(arr, pos+1, high);
// 	}
// 	return;
// }


// performs median filtering
void medianfilt::run()
{
	padd();
	alloc_output();
	const auto tStart = std::chrono::high_resolution_clock::now();
	
	const auto nKernel = kernelSize.x * kernelSize.y * kernelSize.z;
	const auto sizeKernel = nKernel * sizeof(float);
	const auto centerIdx = (nKernel - 1) / 2;

	// temprary array used for sorting
	vector<float> sortArray(nKernel);
	float* localArray = new float [nKernel];

	for (auto iz = 0; iz < dataSize.z; iz++)
	{
		for (auto iy = 0; iy < dataSize.y; iy++)
		{
			// for first element along x we fill the entire array
			for (int xrel = 0; xrel < kernelSize.x; xrel++)
			{
				const int xAbs = xrel;
				for (int zrel = 0; zrel < kernelSize.z; zrel++)
				{
					const int zAbs = iz + zrel;

					const int idxPadd = iy + paddedSize.y * (zAbs + paddedSize.z * xAbs);
					const int idxKernel = kernelSize.y * (zrel + kernelSize.z * xrel);
					memcpy(&localArray[idxKernel], &dataPadded[idxPadd], kernelSize.y * sizeof(float));
					
				}
			}

			memcpy(sortArray.data(), localArray, sizeKernel);
			std::nth_element(sortArray.begin(), sortArray.begin() + nKernel / 2, sortArray.end());
			const int idxOut1 = dataSize.x * (iy + dataSize.y * iz);
			dataOutput[idxOut1] = sortArray[nKernel / 2];

			// now we start overwriting planes of memory
			for (auto ix = 1; ix < dataSize.x; ix++)
			{				
				// current plane we overwrite
				const int xkernel = (ix - 1) % kernelSize.x;
				const int xAbs = ix + kernelSize.x - 1;

				for (int zrel = 0; zrel < kernelSize.z; zrel++)
				{
					const int zAbs = iz + zrel;
					// start index in padded array for copy operation
					const int idxPadd = iy + paddedSize.y * (zAbs + paddedSize.z * xAbs);
					// start index in kernel array for copy operation
					const int idxKernel = kernelSize.y * (zrel + kernelSize.z * xkernel);
					memcpy(&localArray[idxKernel], &dataPadded[idxPadd], kernelSize.y * sizeof(float));
				}

				memcpy(sortArray.data(), localArray, sizeKernel);
				std::nth_element(sortArray.begin(), sortArray.begin() + nKernel / 2, sortArray.end());

				const int idxOut2 = ix + dataSize.x * (iy + dataSize.y * iz);
				dataOutput[idxOut2] = sortArray[nKernel / 2];
			}
		}
	}
	const auto tStop = std::chrono::high_resolution_clock::now();
	const auto tDuration = std::chrono::duration_cast<std::chrono::milliseconds>(tStop- tStart);
	tExec = tDuration.count();

	delete[] localArray;
	return;
}

// creates a padded version of the input volume, order for median: y, z, x
void medianfilt::padd()
{
	printf("Padding of median filter is running\n");
	paddedSize = get_paddedSize();
	alloc_padded();
	for (int iz = 0; iz < paddedSize.z; iz++)
	{
		const bool isZRange = ((iz >= range.z) && (iz <= (paddedSize.z - range.z - 1)));
		for (int iy = 0; iy < paddedSize.y; iy++)
		{
			const bool isYRange = ((iy >= range.y) && (iy <= (paddedSize.y - range.y - 1)));
			for (int ix = 0; ix < paddedSize.x; ix++)
			{
				const bool isXRange = ((ix >= range.x) && (ix <= (paddedSize.x - range.x - 1)));
				// if we are in valid volume, set to value, otherwise padd to 0 for now
				const int idxPad = iy + paddedSize.y * (iz + paddedSize.z * ix);
				if (isZRange && isXRange && isYRange)
				{
					const int idxInput = (ix - range.x) + dataSize.x * ((iy - range.y) + dataSize.y * (iz - range.z));
					// (iz - range.z)
					dataPadded[idxPad] = dataInput[idxInput];
				} 
				else // set 0 for now, later with symmetries etc.
				{
					dataPadded[idxPad] = 0;
				}
			}
		}
	}
	return;
}