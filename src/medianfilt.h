
#include "genfilt.h"
#include "vector3.h"
#include <cstring>
#include <vector>
#include <algorithm>


#ifndef MEDIANFILTSETT_H
#define MEDIANFILTSETT_H

struct medianfiltsett
{
	int kernelSize[3];
};

#endif


#ifndef MEDIANFILT_H
#define MEDIANFILT_H

class medianfilt : public genfilt
{
private:
	void padd();
public:
	void run();
};

#endif