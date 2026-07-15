#include <catch2/catch_test_macros.hpp>

#include "../src/lexer.h"

TEST_CASE("lexer parses a short script without throwing", "[lexer]")
{
	lexer l;
	REQUIRE_NOTHROW(l.parse("c = a * b;\nc = meanfilt(c);\n"));
}

TEST_CASE("lexer splits input into ';'-separated statements", "[lexer]")
{
	// regression: parse() called split_tokens() (which iterates the always-empty
	// splitString) instead of split_lines(), so no statements were ever extracted.
	lexer l;
	l.parse("c = a * b;\nc = meanfilt(c);\n");
	REQUIRE(l.get_nStatements() == 2);

	lexer l2;
	l2.parse("a = 1;");
	REQUIRE(l2.get_nStatements() == 1);
}
