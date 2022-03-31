#include "../src/cudaTools.cuh"

int main()
{
	cudaTools myTools;

	myTools.print_devProps();

	return 0;
}