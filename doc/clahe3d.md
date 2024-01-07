# CLAHE3D

Algorithm description.

## Basic principle of histogram equalization

Medical images usually represent a physical unit and therefore are not necessarily optimized for visual contrast. In CT imaging this means that e.g. the HU values for the liver and the surrounding organs are in a very close range while bone and air span a large distance on the scale. Setting the gray-scale limits to the maximum and minimum of a slice therefore does not allow to visualize e.g. soft tissues with contrast. Most slicing software therefore has so called pre-defined windows to provide contrast for different tissues such as liver. Nevertheless, those windows will lead to a saturation of a large range of other organs and hence only provide contrast for a specific region. Disadvantageously, the visible range is not used well since e.g. almost no structures are visible in the range between -900 and -100 and +500 to +1000 wasting the capacity to see the difference in HU values of different organs in a single image.

Contrast enhancement algorithms have been introduced in the past to fight those shortcomings. They are usually based on the principle of equalizing the histogram of the image. The histogram is defined as the distribution of intensities in an image and will show peaks around the HU value of the organs. An optimal histogram would be flat, distributing the relevant content of the image over the entire visible range of the colorbar. A histogram equalization algorithm therefore takes an image with a non-uniform histogram as an input and converts it into an image with a uniform contrast by mapping the physical values in the image to a non-linear distribution in the output of the image.

The basic methodology can be described as follows: 

- calculate the histogram of the image or image region 
- normalize the histogram so that the sum of all bins equals 1.0
- calculate the cumulative distribution function (CDF) which is defined as the sum of all previous bins
- use the inverse of the cumulative distribution function (iCDF) to map each value in the input image to a value in the 0.0 to 1.0 range 

After applying the principle described above, the image has a uniform contrast in the output image and the colorbar applied to it can be used in a meaningful way.

## Adaptive histogram equalization

Adaptive histogram equalization is taking things a step further. Instead of using the same inversion function for the entire image, the image is subdivided into different regions and the above routine is applied to each region individually. This means that within the liver, the lower value will be represented by the lowest colorbar limit and the highest value by the highest colorbar limit allowing good differentiation of contrast within the liver itself whereas in bony structures, those limits will be defined to match the max and min of the bone itself. This allows high contrast in different neighborhoods at the cost of grey-scale values that have different physical representation over the image.

To avoid tiling artifacts, the histograms will be calculated for overlapping sub-regions and the inversion function for a pixel is interpolated from the neighboring sub-volumes to allow smooth transitions in the image and avoid edge effects.

## Moving to volumetric datasets

As a last step, this procedure can not only be applied to 2D image but to 3D volumes. Instead of calculating the histogram of sub-images, local sub-volumes are used to determine the iCDF and then 3D linear interpolation is used to calculate the mapping from physical values (e.g. HU) to colorbar values. However, this algorithm comes at the cost of a high computational complexity which is why the algorithm was implement both for CPU and GPU.

## Scripted version of tool

There is an executable provided with the toolbox that allows running the algorithm from the command line. The tool thereby takes 6 input parameters

- spacing between the sub-volumes used for calculation of the local histograms in voxel
- size of the sub-volumes along all three dimensions in voxel
- noise level defining the lower clip limit of the histogram to avoid e.g. contrast enhancement in air
- number of bins of the histogram
- path to the input file
= path to the output file

For a CT dataset, a useful set of starting values would be:

```bash
./clahe3d_scripted 9 21 -950.0 255 CT.nii CT_clahe.nii
```