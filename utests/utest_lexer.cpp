#include <catch2/catch.hpp>
#include "../src/lexer.h"

TEST_CASE("lexer basic parsing", "[lexer]")
{
	lexer l;
	l.parse("c = a * b;\nc = meanfilt(c);\n");
}