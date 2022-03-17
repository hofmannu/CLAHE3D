#include "../src/lexer.h"

int main()
{

	lexer l;
	l.parse("c = a * b;\nc = meanfilt(c);\n");

	return 0;
}