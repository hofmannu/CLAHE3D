
#include "genfilt.h"

#ifndef MEDIANFILTSETT_H
#define MEDIANFILTSETT_H

struct medianfiltsett
{
	int kernelSize[3];
};

#endif


#ifndef MEDIANFILT_H
#define MEDIANFILT_H

class medianfilt : public medianfiltsett, public genfilt
{
private:

public:
	void run();
};

#endif