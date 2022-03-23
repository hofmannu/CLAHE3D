/*
	log creation tool
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 17.03.2022
	Description: Simple class which generates a log

	Changelog:
		- added flags allowing simple filtering of certain entry types such as warnings

*/

#include <string>
#include <vector>
#include <chrono>

#ifndef LOGENTRY_H
#define LOGENTRY_H

struct logentry
{
	int type; // 0 - normal entry, 1 - warning, 2 - error
	std::string message; // the message passed to our log system from volproc
	std::string tString; // timepoint of log message as a string
	std::time_t timeStamp; // timestamp when log message arrived
};

#endif

#ifndef LOG_H
#define LOG_H

enum log_type {LOG, WARNING, ERROR};

class log
{
	std::string fullLog;
	std::vector<logentry> collection;

	// defines which logs will be returned
	bool flagLog = 1;
	bool flagWarning = 1;
	bool flagError = 1;


	bool isEntryWanted(const logentry& entry) const;

public:
	void new_general(const std::string message, const int type);
	void new_entry(const std::string message);
	void new_warning(const std::string warning);
	void new_error(const std::string warning);
	void clear_log();

	std::string get_log_string() const;

	const std::vector<logentry> get_log() const;

	// returns a pointer to our exclusion flags
	bool* get_pflagLog() {return &flagLog;};
	bool* get_pflagWarning() {return &flagWarning;};
	bool* get_pflagError() {return &flagError;};
};

#endif