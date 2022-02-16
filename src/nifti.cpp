#include "nifti.h"

#define MIN_HEADER_SIZE 348
#define NII_HEADER_SIZE 352

nifti::~nifti()
{
  if (isDataMatrixAlloc)
    delete[] dataMatrix;
}

// read_nifti_file
int nifti::read_header(const string _filePath)
{
  filePath = _filePath;
 
  // open and read header
  FILE *fp = fopen(filePath.c_str(), "r");
  if (fp == NULL) 
  {
    printf("Error opening header file %s\n", filePath.c_str());
    throw "FileError";
  }

  int ret = fread(&hdr, MIN_HEADER_SIZE, 1, fp);
  if (ret != 1) 
  {
    printf("Error reading header file %s\n", filePath.c_str());
    throw "OperationFailed";
  }
  fclose(fp);
  return 0;
}

void nifti::print_header()
{
  /********** print a little header information */
  printf("\n%s header information:", filePath.c_str());
  printf("\nXYZT dimensions: %d %d %d %d",
    hdr.dim[1], hdr.dim[2], hdr.dim[3], hdr.dim[4]);
  printf("\nDatatype code and bits/pixel: %d %d",
    hdr.datatype, hdr.bitpix);
  printf("\nScaling slope and intercept: %.6f %.6f",
    hdr.scl_slope, hdr.scl_inter);
  printf("\nByte offset to data in datafile: %ld",
    (long) (hdr.vox_offset));
  return;
}

// allocates memory for incoming data 
void nifti::alloc_mem()
{
  if (isDataMatrixAlloc)
  {
    delete[] dataMatrix;
  }
  dataMatrix = new float [hdr.dim[1]*hdr.dim[2]*hdr.dim[3]];
  isDataMatrixAlloc = 1;
  return;
}

int nifti::read_data(const string _filePath)
{
  filePath = _filePath;
  read_data();
  return 0;
}

int nifti::read_data()
{
  alloc_mem();


  // open the datafile, jump to data offset
  FILE *fp = fopen(filePath.c_str(),"r");
  if (fp == NULL) 
  {
    printf("Error opening data file %s\n", filePath.c_str());
    exit(1);
  }

  int ret = fseek(fp, (long)(hdr.vox_offset), SEEK_SET);
  if (ret != 0) 
  {
    printf("Error doing fseek() to %ld in data file %s\n",
      (long)(hdr.vox_offset), filePath.c_str());
    exit(1);
  }

  if (hdr.datatype == DT_FLOAT)
  {
    ret = fread(dataMatrix, sizeof(float), hdr.dim[1] * hdr.dim[2] * hdr.dim[3], fp);
    if (ret != hdr.dim[1] * hdr.dim[2] * hdr.dim[3])
    {
      printf("Error reading volume 1 from %s (%d)\n",
        filePath.c_str(), ret);
      exit(1);
    }
  }
  else
  {
    printf("Data type requires implementation!\n");
    throw "InvalidValue";
  }

  fclose(fp);

  // scale the data buffer
  if (hdr.scl_slope != 0) 
  {
    for (int i=0; i < hdr.dim[1] * hdr.dim[2] * hdr.dim[3]; i++)
      dataMatrix[i] = (dataMatrix[i] * hdr.scl_slope) + hdr.scl_inter;
  }

  // print mean of data, get min, get max
  minVal = dataMatrix[0]; 
  maxVal = dataMatrix[0];
  double total = 0;
  for (int i = 0; i < hdr.dim[1] * hdr.dim[2] * hdr.dim[3]; i++)
  {
    total += dataMatrix[i];
    if (dataMatrix[i] > maxVal)
      maxVal = dataMatrix[i];
    if (dataMatrix[i] < minVal)
      minVal = dataMatrix[i];
  }
  total /= (hdr.dim[1] * hdr.dim[2] * hdr.dim[3]);
  printf("Mean of volume 1 in %s is %.3f\n", 
    filePath.c_str(), total);


  return 0;
}
