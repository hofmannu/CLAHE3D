
#ifndef LEXER_H
#define LEXER_H

#include <iostream>
#include <string>
#include <cctype>
#include <algorithm>
#include <vector>

class lexer
{
private:
	std::string inputString;
	std::vector<std::string> splitString;
	std::vector<std::vector<std::string>> tokenList;

	void remove_whitespace();
	void split_lines();
	void split_tokens();

public:
	void parse(const std::string inputString);
	void print_result() const;
};

#endif