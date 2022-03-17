#ifndef LEXER_H
#define LEXER_H

#include <iostream>
#include <string>
#include <cctype>
#include <algorithm>

using namespace std;

class lexer
{
private:
	string inputString;
	vector<string> splitString;
	vector<vector<string>> tokenList;

	void remove_whitespace();
	void split_lines();
	void split_tokens();

public:
	void parse(const std::string inputString);
	void print_result() const;
};

#endif