
% recompile our code first
mex -g -O clahe3d.cpp interpGrid.cpp histeq.cpp

% generate required testing variables
inputVol = rand(600, 500, 400);
inputVol = single(inputVol);
subVolSize = uint64([10, 10, 10]);
clipLimit = single(0.5);
binSize = uint64(100);

% test code
clahe3d(inputVol, subVolSize, clipLimit, binSize);

