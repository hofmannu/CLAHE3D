#include <string>
#include <vector>
#include <chrono>


#ifndef LOGENTRY_H
#define LOGENTRY_H

struct logentry
{
	int type; // 0 - normal entry, 1 - warning, 2 - error
	std::string message;
	std::string tString;
	std::time_t timeStamp;
};

#endif

#ifndef LOG_H
#define LOG_H


class log
{
	std::string fullLog;
	std::vector<logentry> collection;
public:
	void new_general(const std::string message, const int type);
	void new_entry(const std::string message);
	void new_warning(const std::string warning);
	void new_error(const std::string warning);
	void clear_log();

	std::string get_log_string() const;

	const std::vector<logentry> get_log() const;
};

#endif