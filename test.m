
% recompile our code first
cd src
mex -g -O clahe3dmex.cpp interpGrid.cpp histeq.cpp
cd ..

addpath(genpath(pwd))

% generate required testing variables
load('/media/hofmannu/r_sync/hfoam_raw_data/08_MouseWounds/AR/2019_11_21/005_mouseBackMidRough_578_safted.mat')

saftedVol = single(saftedVol);
saftedVol = abs(saftedVol);
saftedVol = saftedVol ./ max(saftedVol(:));

binSizeMm = [0.5, 4, 4] * 1e-3;
subVolSize = uint64(binSizeMm ./ [Settings.z(2) - Settings.z(1), Settings.dX, Settings.dY])

clipLimit = single(0.01);
binSize = uint64(1000);
% vtkwrite('/media/hofmannu/hofmannu/unclahed.vtk', 'structured_points', 'unclahed', saftedVol, 'spacing', Settings.z(2) - Settings.z(1), Settings.dX, Settings.dY, 'BINARY');

% test code
clahe3dmex(saftedVol, subVolSize, clipLimit, binSize);

nIFin = sum(~isfinite(saftedVol(:)));
nTotal = length(saftedVol(:));
nIFin / nTotal * 100

saftedVol(~isfinite(saftedVol)) = 0;
saftedVol = saftedVol ./ max(saftedVol(:));

vtkwrite('/media/hofmannu/hofmannu/clahed.vtk', 'structured_points', 'clahed', saftedVol, 'spacing', Settings.z(2) - Settings.z(1), Settings.dX, Settings.dY, 'BINARY');

