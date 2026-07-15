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
git submodule update --init --recursive
```

The third-party C/C++ libraries (`imgui` on the docking branch, `glfw` and the
`hdf5` C++ API) are pulled and built by [Conan](https://conan.io/).
Two dependencies that are not on ConanCenter, `ImPlot` and `ImGuiFileDialog`, are
fetched from git and compiled against the Conan `imgui` so their
`ImGuiContext` layout matches the docking build.

What still has to come from your system:

- Build tooling: `git`, `cmake` (>= 3.23 for presets), `g++`, and `conan` (v2)
- GPU support: CUDA toolkit
- GUI support: system OpenGL and X11 runtime/dev libraries

Install Conan (once) with `pipx install conan` and, on a fresh machine, create a
default profile with `conan profile detect`.

# Installation / Compiling

You can use the code also with GPU support and the functions will then be accelerated on the GPU (requires CUDA and CUDA capable device). 
To compile with GPU support, change the flag `USE_CUDA` in the main `CMakeLists.txt` to `TRUE`. 
Don't forget to add the architecture required for your GPU in the `CMakeLists.txt` according to your target hardware. 
If you want to use the [ImGui](https://github.com/ocornut/imgui) based graphical user interface (basically a simple slicing and execution interface), there is also an option for that. 

For a quick build, use the helper script (defaults to `Debug`):

```bash
./build.sh            # Debug build into Debug/
./build.sh Release    # optimised build into Release/
```

Or run the steps manually. The first lets Conan resolve/build the dependencies
and emit a CMake toolchain into `Debug/`; the second configures and builds
against it:

```bash
conan install . --output-folder=Debug --build=missing \
  -s build_type=Debug -s compiler.cppstd=20
cd Debug
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug
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
