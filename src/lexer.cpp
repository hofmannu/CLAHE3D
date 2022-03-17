#include "lexer.h"

// removes all unnecessary whitespace from vector
void lexer::remove_whitespace()
{
	inputString.erase(std::remove_if(inputString.begin(), inputString.end(), ::isspace), inputString.end());
	return;
}

// splits single long string into several substrings
void lexer::split_lines()
{
	splitString.clear();
	// bool done = 0;
	while(1)
	{
		std::size_t pos = inputString.find(";");
		
		if (pos == std::string::npos)
			break;
		string subString = inputString.substr(0, pos);
		splitString.push_back(subString);
		inputString.erase(0, pos + 1);
		// cout << subString << endl;
	}
	return;
}

void lexer::split_tokens()
{
	for (string currLine: splitString)
	{

		cout << currLine << endl;
	}
}

// extract all operators and 

void lexer::parse(const std::string _inputString)
{
	cout << "input string:\n" << _inputString << endl;
	inputString = _inputString;

	// removing all whitespace
	remove_whitespace();
	cout << "after whitespace removal: " << inputString << endl;

	// divide into substrings (for each semicolon one line)
	split_tokens();

	return;
}


void lexer::print_result() const
{

	return;
}
