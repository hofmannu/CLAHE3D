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
	return std::string(buffer);
}

void log::new_general(const std::string message, const int type)
{
	logentry newEntry;
	newEntry.message = message;
	newEntry.type = type;
	newEntry.timeStamp = get_tStamp();
	newEntry.tString = get_tString(newEntry.timeStamp);
	collection.push_back(newEntry);
	return;
}

void log::new_entry(const std::string message)
{
	new_general(message, 0);
	return;
}

void log::new_warning(const std::string message)
{
	new_general(message, 1);
	return;
}

void log::new_error(const std::string message)
{
	new_general(message, 2);
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

// returns the full vector containing all our logs filtered according to our flags
const std::vector<logentry> log::get_log() const
{
	std::vector<logentry> returnVec;
	for (logentry elem: collection)
	{
		if (elem.type == 0)
		{
			if (flagLog) returnVec.push_back(elem);
		}
		else if (elem.type == 1)
		{
			if (flagLog) returnVec.push_back(elem);
		}
		else if (elem.type == 2)
		{
			if (flagLog) returnVec.push_back(elem);
		}
	}
	return returnVec;
}

