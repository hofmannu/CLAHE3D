# CLAHE3D

A three dimensional contrast enhancement code written in `C++` and `CUDA`.
Works as standalone C++ application or through a MATLAB interface specified as a `*.mex` file.
Histograms are calculated for subvolumes of definable size. The spacing of the histogram bins can be chosen independently of the bin size.
Afterwards, Each histogram is then converted into a normalized cummulative distribution function.
By interpolating the inverted distribution function for each subregion, we can enhance the local contrast in a volumetric image.

Beside the basic functionality of CLAHE3D, I started implementing a few more volume processing functions which can improve the outcome of CLAHE3D such as

-  mean filtering
-  gaussian filtering
-  median filtering
-  thresholding of volumes (similar to clip limit)
-  normalization to custom range

All operations support multithreading on CPUs.
Some are already pushed to the GPU, others not yet. 

![Preview of the effect CLAHE3D has on a medical volume](https://hofmannu.org/assets/clahe3d.9gMm5Hvc.png)

# Cloning and Dependencies

```bash
git clone https://github.com/hofmannu/CLAHE3D.git
cd CLAHE3D
git submodule init
git submodule update
```

To use the GUI or CUDA support, there are a few more dependencies to install

## Ubuntu

- Compilation tools and helpers: `git`, `cmake`, `g++`
- GPU: `nvidia-cuda-toolkit`
- Stuff to display things using the GUI: `libglfw3-dev`
- Saving and reading from h5 files: `libhdf5-dev`

With those libraries installed it should work. A few libraries might not be matching with your OS path to them (e.g. not finding header files). The code was only tested for ArchLinux to its full extend.

## ArchLinux

- Compilation tools and helpers: `git`, `cmake`, `g++`
- GPU: `cuda`
- Stuff to display things using the GUI: `glfw-x11` or `glfw-wayland`, `glew`
- Saving and reading from h5 files: `hdf5`

# Installation / Compiling

You can use the code also with GPU support and the functions will then be accelerated on the GPU (requires CUDA and CUDA capable device). 
To compile with GPU support, change the flag `USE_CUDA` in the main `CMakeLists.txt` to `TRUE`. 
Don't forget to add the architecture required for your GPU in the `CMakeLists.txt` according to your target hardware. 
If you want to use the [ImGui](https://github.com/ocornut/imgui) based graphical user interface (basically a simple slicing and execution interface), there is also an option for that. 

Execute the following commands from the project directory to build the software:

```bash
mkdir Debug
cd Debug 
cmake ..
make all
```

To run a throughout test of the procedures, run `ctest` from the `Debug` subfolder after compilation.

![Preview of GUI](https://hofmannu.org/assets/clahe-gui.ByeVTKPz.png)

# Feature request / bug report

Feel free to contact me if you need the addition of a feature or if you want to report a bug.
Next step of this project is to enable the simultaneous processing of multiple volume pipelines using a simple scripting language.
Other feature requests and processing functions are listed in the issues section.
Furthermore, the slicer should get some advanced features soon.

# Literature about CLAHE3D
- Karen Lucknavalai, Jürgen P. Schulze: [Real-Time Contrast Enhancement for 3D Medical Images Using Histogram Equalization](https://link.springer.com/chapter/10.1007/978-3-030-64556-4_18) in ISVC 2020: Advances in Visual Computing pp 224-235 --> describes a layer wise processing but does not use volumetric tiling
- P Amorim et al.: [3D Adaptive Histogram Equalization Method for Medical Volumes.](https://www.scitepress.org/Papers/2018/66153/66153.pdf) in VISIGRAPP (4: VISAPP), 2018
- V. Stimper et al.: [Multidimensional Contrast Limited Adaptive Histogram Equalization](https://ieeexplore.ieee.org/abstract/document/8895993)
