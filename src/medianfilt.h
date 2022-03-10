
#include "genfilt.h"
#include "vector3.h"

#ifndef MEDIANFILTSETT_H
#define MEDIANFILTSETT_H

struct medianfiltsett
{
	int kernelSize[3] = {3, 3, 3};
};

#endif


#ifndef MEDIANFILT_H
#define MEDIANFILT_H

class medianfilt : public genfilt
{
private:

public:
	void run();
};

#endif