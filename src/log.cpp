#include "log.h"

std::time_t get_tStamp()
{
	std::chrono::time_point<std::chrono::system_clock> curr;
	curr = std::chrono::system_clock::now();;
	std::time_t currTime = std::chrono::system_clock::to_time_t(curr);
	return currTime;
}

void log::new_entry(const std::string message)
{
	logentry newEntry;
	newEntry.message = message;
	newEntry.type = 0;
	newEntry.timeStamp = get_tStamp();
	collection.push_back(newEntry);
	return;
}

void log::new_warning(const std::string message)
{
	logentry newEntry;
	newEntry.message = message;
	newEntry.type = 1;
	newEntry.timeStamp = get_tStamp();
	collection.push_back(newEntry);
	return;
}

void log::new_error(const std::string message)
{
	logentry newEntry;
	newEntry.message = message;
	newEntry.type = 2;
	newEntry.timeStamp = get_tStamp();
	collection.push_back(newEntry);
	return;
}

// emties the entire log
void log::clear_log()
{
	collection.clear();
	return;
}

std::string log::get_log() const 
{
	std::string outputMessage = "";
	for (int iLog = 0; iLog < collection.size(); iLog++)
	{
		outputMessage += std::ctime(&collection[iLog].timeStamp);
		outputMessage += " -> " + collection[iLog].message + "\n";
	}
	return outputMessage;
}

