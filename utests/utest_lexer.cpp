#include <catch2/catch_test_macros.hpp>

#include "../src/lexer.h"

TEST_CASE("lexer parses a short script without throwing", "[lexer]")
{
	lexer l;
	REQUIRE_NOTHROW(l.parse("c = a * b;\nc = meanfilt(c);\n"));
}
