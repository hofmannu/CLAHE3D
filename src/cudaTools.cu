#include "cudaTools.cuh"

cudaTools::cudaTools() {
  // get number of connected devices
  cudaError_t err = cudaGetDeviceCount(&nDevices);
  if (err != cudaSuccess) {
    printf("Could not receive a device count\n");
    throw "CudaError";
  }

  // query device properties for each attached GPU
  cudaDeviceProp currProps;
  for (int iDevice = 0; iDevice < nDevices; iDevice++) {
    err = cudaGetDeviceProperties(&currProps, iDevice);
    if (err != cudaSuccess) {
      printf("Something went wrong while returning properties of device %d\n",
             iDevice);
      throw "CudaError";
    }

    props.push_back(std::move(currProps));
  }
  return;
}

void cudaTools::print_devProps() {
  for (int iDevice = 0; iDevice < nDevices; iDevice++)
    print_devProps(iDevice);
  return;
}

// prints the properties of a device to the terminal
void cudaTools::print_devProps(const int iDevice) {
  if (iDevice < 0) {
    printf("Device index must be larger or equal 0!\n");
    throw "InvalidValue";
  }

  if (iDevice >= nDevices) {
    printf("Requested ID is exceedin available device number\n");
    throw "InvalidValue";
  }

  printf("General information for device %d: \n", iDevice);
  printf(" - Name: %s\n", props[iDevice].name);
  printf(" - Compute capability: %d.%d\n", props[iDevice].major,
         props[iDevice].minor);

  printf(" - Total global mem: %ld bytes\n", props[iDevice].totalGlobalMem);
  printf(" - Total constant mem: %ld bytes\n", props[iDevice].totalConstMem);
  printf(" - Max mem pitch: %ld\n", props[iDevice].memPitch);
  printf(" - Texture alignment: %ld\n", props[iDevice].textureAlignment);
  printf(" - Multiprocessor count: %d\n", props[iDevice].multiProcessorCount);
  printf(" - Shared memory per MP: %d bytes\n",
         props[iDevice].sharedMemPerBlock);
  printf(" - Registers per MP: %d\n", props[iDevice].regsPerBlock);
  printf(" - Threads in warp: %d\n", props[iDevice].warpSize);
  printf(" - Max threads per block: %d\n", props[iDevice].maxThreadsPerBlock);
  printf(" - Max thread dimensions: (%d, %d, %d)\n",
         props[iDevice].maxThreadsDim[0], props[iDevice].maxThreadsDim[1],
         props[iDevice].maxThreadsDim[2]);
  printf(" - Max grid dimensions: (%d, %d, %d)\n",
         props[iDevice].maxGridSize[0], props[iDevice].maxGridSize[1],
         props[iDevice].maxGridSize[2]);

  printf("\n");
  return;
}

void cudaTools::checkCudaErr(const cudaError_t &err, const char *msgErr) {
  if (err != cudaSuccess) {
    printf("There was some CUDA error: %s, %s\n", msgErr,
           cudaGetErrorString(err));
    throw "CudaError";
  }
  return;
}
