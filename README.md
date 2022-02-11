# CLAHE3D

A three dimensional contrast enhancement code written in `C++`. Works as standalone C++ application or through a MATLAB interface specified as a `*.mex` file. Since I am using it at the moment only through `mex` I did not implement any file import or export functions for the `C++` based version but can do that upon reasonable request. Histograms are calculated for subvolumes and afterwards interpolated to the currently adapted pixel through trilinear interpolation.

# MATLAB
Change to folder and run `mex -O clahe3dmex.cpp interpGrid.cpp histeq.cpp`. Afterwards you can use the function as:
`clahe3dmex(interpVol, subVolSize, spacingSubVols, clipLimit, binSize);`
where 

*  `vol` is the three dimensional volume of type `single`,
*  `subVolSize` is the container size used to create the histograms subvolume size specified as a 3 element vector of type `uint64`
*  `spacingSubVols` is the spacing vector between the histograms specified as `uint64`
*  `clipLimit` is the clipping threshold specified as `single`
*  and `binSize` is the number of histogram bins specified as `uint64`. 

Tested using `MATLAB 2019 A` and `MATLAB 2019 B`. The function `utest_bone.m` serves as an illustration.

# C++

The code is compiled using `cmake`.

```bash
cd Debug
cmake ..
make all
./main_exp
```

An example of how to use the main class can be found in `src/main.cpp`.

# Feature request / bug report
Feel free to contact me if you need the addition of a feature or if you want to report a bug. Next step of this project is going to be a `CUDA` capable parallelized version of the code.
