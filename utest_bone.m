
% recompile our code first
% cd src
% mex -g -O -v clahe3dmex.cpp histeq.cpp gridder.cpp
% cd ..

addpath(genpath(pwd))
load ct.mat

[nz, nx, ny] = size(interpVol); 

subVolSize = uint64([21, 21, 21]);
spacingSubVols = uint64([5, 5, 5]);

clipLimit = single(1);
binSize = uint64(255);
% vtkwrite('/media/hofmannu/hofmannu/unclahed.vtk', 'structured_points', 'unclahed', saftedVol, 'spacing', Settings.z(2) - Settings.z(1), Settings.dX, Settings.dY, 'BINARY');

% test code
oldSlice = interpVol(100, :, :);
oldSlice = reshape(oldSlice, [nx, ny]);

clahe3dmex(interpVol, subVolSize, spacingSubVols, clipLimit, binSize);

nIFin = sum(~isfinite(interpVol(:)));
if (nIFin > 0)
	error("Function returned invalid values");
end

newSlice = interpVol(100, :, :);
newSlice = reshape(newSlice, [nx, ny]);

figure()
subplot(1, 2, 1)
imagesc(oldSlice);
axis image;
title("Before");

subplot(1, 2, 2)
imagesc(newSlice);
axis image;
title("After");

colormap(bone)
