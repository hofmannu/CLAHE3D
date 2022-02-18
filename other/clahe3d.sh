#!/bin/bash


# example usage:
# ./clahe3d.sh 10 21 -400 255 CT.nii CT_clahed.nii

sv_spacing=$1
sv_size=$2
clip_value=$3
bin_size=$4
nifti_in=$5
nifti_out=$6

./../Debug/clahe3d_scripted $sv_spacing $sv_size $clip_value $bin_size $nifti_in $nifti_out
