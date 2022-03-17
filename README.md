# CLAHE3D

A three dimensional contrast enhancement code written in `C++` and `CUDA`. Works as standalone C++ application or through a MATLAB interface specified as a `*.mex` file. Histograms are calculated for subvolumes of definable size. The spacing of the histogram bins can be chosen independently of the bin size. Afterwards, Each histogram is then converted into a normalized cummulative distribution function. By interpolating the inverted distribution function for each subregion, we can enhance the local contrast in a volumetric image.

Beside the basic functionality of CLAHE3D I started implementing a few more volume processing functions which can improve the outcome of CLAHE3D such as

*  meanfiltering of volumes running on multiple cores on the CPU
*  gaussian filtering
*  thresholding of volumes (similar to clip limit)
*  normalization to custom range 

File formats supported for reading so far are

*  `nii`
*  `h5`

![Preview of the effect CLAHE3D has on a medical volume](https://hofmannu.org/wp-content/uploads/2022/03/clahe3d-768x406.png)

# Installation / Compiling

The `CPU` code does not depend on any library. For the `GPU` version you need to have `CUDA` installed on your system. For the GUI there come a few additional dependencies to show the images.

# MATLAB
Change to folder the subfolder `src` and run `mex -O clahe3dmex.cpp gridder.cpp histeq.cpp`. Afterwards you can use the function as:
`clahe3dmex(interpVol, subVolSize, spacingSubVols, clipLimit, binSize);`
where 

*  `vol` is the three dimensional volume of type `single`,
*  `subVolSize` is the container size used to create the histograms subvolume size specified as a 3 element vector of type `uint64`
*  `spacingSubVols` is the spacing vector between the histograms specified as `uint64`
*  `clipLimit` is the clipping threshold specified as `single`
*  and `binSize` is the number of histogram bins specified as `uint64`. 

Tested using `MATLAB 2019 A` and `MATLAB 2019 B`. The function `utest_bone.m` serves as an illustration.

# C++

The code is compiled using `cmake`. You can use the code also with GPU support and the functions will then be accelerated on the GPU (requires CUDA and CUDA capable device). To compile with GPU support, change the flag `USE_CUDA` in the main `CMakeLists.txt` to `TRUE`.

```bash
mkdir Debug
cd Debug 
cmake ..

make all
ctest
```

An example of how to use the main class can be found in `src/main.cpp`. To run a throughout test of the procedures, run `ctest` from the `Debug` subfolder after compilation. The software ships with an ImGUI based user interface.

![Preview of GUI](https://hofmannu.org/wp-content/uploads/2022/03/Screenshot_2022-03-10_16-15-58-768x426.png)


# Feature request / bug report

Feel free to contact me if you need the addition of a feature or if you want to report a bug. Next step of this project is going to be a `CUDA` capable parallelized version of the code.


# Literature
*  Karen Lucknavalai, Jürgen P. Schulze: [Real-Time Contrast Enhancement for 3D Medical Images Using Histogram Equalization](https://link.springer.com/chapter/10.1007/978-3-030-64556-4_18) in ISVC 2020: Advances in Visual Computing pp 224-235 --> describes a layer wise processing but does not use volumetric tiling
*  P Amorim et al.: [3D Adaptive Histogram Equalization Method for Medical Volumes.](https://www.scitepress.org/Papers/2018/66153/66153.pdf) in VISIGRAPP (4: VISAPP), 2018
*  V. Stimper et al.: [Multidimensional Contrast Limited Adaptive Histogram Equalization](https://ieeexplore.ieee.org/abstract/document/8895993)
