#include "nifti.h"

#define MIN_HEADER_SIZE 348
#define NII_HEADER_SIZE 352

nifti::~nifti()
{
  if (isDataMatrixAlloc)
    delete[] dataMatrix;
}

// read_nifti_file


int nifti::read(const string _filePath)
{

  filePath = _filePath;
  read();
  return 0;
}

int nifti::read()
{

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

  ret = fseek(fp, (long)(hdr.vox_offset), SEEK_SET);
  if (ret != 0) 
  {
    printf("Error doing fseek() to %ld in data file %s\n",
      (long)(hdr.vox_offset), filePath.c_str());
    throw "InvalidOperation";
  }

  alloc_mem();

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
  else if (hdr.datatype == DT_INT16)
  {
    int16_t* tempArray = new int16_t[hdr.dim[1] * hdr.dim[2] * hdr.dim[3]];
    ret = fread(tempArray, sizeof(int16_t), hdr.dim[1] * hdr.dim[2] * hdr.dim[3], fp);
    if (ret != hdr.dim[1] * hdr.dim[2] * hdr.dim[3])
    {
      printf("Error reading volume 1 from %s (%d)\n",
        filePath.c_str(), ret);
      exit(1);
    }

    // convert to float
    for (int iElem = 0; iElem < (hdr.dim[1] * hdr.dim[2] * hdr.dim[3]); iElem++)
      dataMatrix[iElem] = (float) tempArray[iElem];

    delete[] tempArray;
  }
  else
  {
    printf("Data type %d requires implementation!\n", hdr.datatype);
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

  return 0;
}

void nifti::save(const string _filePath)
{
  nifti1_extender pad={0,0,0,0};
  int ret, i;

  hdr.datatype = DT_FLOAT;
  
  FILE *fp = fopen(_filePath.c_str(), "w");
  if (fp == NULL) 
  {
    printf("Error opening header file %s for write\n", filePath.c_str());
    throw "FileError";
  }

  ret = fwrite(&hdr, MIN_HEADER_SIZE, 1, fp);
  if (ret != 1) 
  {
    printf("Error writing header file %s\n", filePath.c_str());
    throw "FileError";
  }

  ret = fwrite(&pad, 4, 1, fp);
  if (ret != 1) 
  {
    printf("Error writing header file extension pad %s\n", filePath.c_str());
   throw "FileError";
  }

  ret = fwrite(dataMatrix, sizeof(float), hdr.dim[1] * hdr.dim[2] * hdr.dim[3] * hdr.dim[4], fp);
  if (ret != hdr.dim[1] * hdr.dim[2] * hdr.dim[3] * hdr.dim[4]) {
    printf("Error writing data to %s\n", filePath.c_str());
    throw "FileError";
  }

  fclose(fp);

  return;
}

void nifti::print_header()
{
  /********** print a little header information */
  printf("\n%s header information:", filePath.c_str());
  printf("\nXYZT dimensions: %d %d %d %d",
    hdr.dim[1], hdr.dim[2], hdr.dim[3], hdr.dim[4]);
  printf("\nDatatype code and bits/pixel: %d %d",
    hdr.datatype, hdr.bitpix);
  printf("Scaling slope and intercept: %.6f %.6f\n",
    hdr.scl_slope, hdr.scl_inter);
  printf("Byte offset to data in datafile: %ld\n",
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

float nifti::get_val(const vector3<int> pos) const
{
  return dataMatrix[pos.x + hdr.dim[1] * (pos.y + hdr.dim[2] * pos.z)];
}