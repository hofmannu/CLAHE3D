#include <string>
#include <vector>
#include "../lib/CVolume/src/volume.h"


#ifndef WORKSPACE_H
#define WORKSPACE_H

struct workspace_entry
{
	std::string name;
	std::size_t id;
	volume vol;
};

struct entry_info
{
	std::size_t memBytes;
	std::size_t dim[3];
	float res[3];
	float origin[3];
};

class workspace
{

private:
	std::vector<workspace_entry> entries;

public:
	void add_entry(const string name);
	void add_entry(const string name, const volume& newVol);
};

#endif