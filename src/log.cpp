#include "log.h"

std::time_t get_tStamp()
{
	std::chrono::time_point<std::chrono::system_clock> curr;
	curr = std::chrono::system_clock::now();;
	std::time_t currTime = std::chrono::system_clock::to_time_t(curr);
	return currTime;
}

std::string get_tString(const std::time_t tStamp)
{
	char buffer [80];
	struct tm * timeinfo = localtime (&tStamp);
	strftime (buffer, 80, "%I:%M:%S %p", timeinfo);
	return std::move(std::string(buffer));
}

void log::new_general(const std::string message, const int type)
{
	logentry newEntry;
	newEntry.message = message;
	newEntry.type = type;
	newEntry.timeStamp = get_tStamp();
	newEntry.tString = get_tString(newEntry.timeStamp);
	collection.push_back(std::move(newEntry));
	return;
}

// adds a normal log entry
void log::new_entry(const std::string message)
{
	new_general(std::move(message), LOG);
	return;
}

// adds a new warning to our journal
void log::new_warning(const std::string message)
{
	new_general(std::move(message), WARNING);
	return;
}

// adds a new error log to our journal
void log::new_error(const std::string message)
{
	new_general(std::move(message), ERROR);
	return;
}

// emties the entire log
void log::clear_log()
{
	collection.clear();
	return;
}

std::string log::get_log_string() const 
{
	std::string outputMessage = "";
	for (int iLog = 0; iLog < collection.size(); iLog++)
	{
		outputMessage += collection[iLog].tString;
		outputMessage += " -> " + collection[iLog].message + "\n";
	}
	return outputMessage;
}

// helper functions to check if element is wanted
bool log::isEntryWanted(const logentry& entry) const
{
	bool isWanted = 0;
	if (entry.type == LOG)
	{
		if (flagLog)
		{
			isWanted = 1;
		}
	}
	else if (entry.type == WARNING)
	{
		if (flagWarning)
			isWanted = 1;
	}
	else if (entry.type == ERROR)
	{
		if (flagError)
			isWanted = 1;
	}
	else
	{
		printf("An invalid argument was passed to this function\n");
		throw "InvalidArgument";
	}
	return isWanted;
}

// returns the full vector containing all our logs filtered according to our flags
const std::vector<logentry> log::get_log() const
{
	// TODO count the number of elements and preallocate

	std::vector<logentry> returnVec;
	for (logentry elem: collection)
	{
		if (isEntryWanted(elem)) 
			returnVec.push_back(elem);
		
	}
	return returnVec;
}

