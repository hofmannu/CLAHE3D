#include <catch2/catch.hpp>
#include "../src/vector3.h"

TEST_CASE("vector3 operations", "[vector3]")
{
	srand(1);

	vector3<int> myVec;

	SECTION("initialization with curly brackets")
	{
		const vector3<int> curlyVector = {12, 13, 14};
		REQUIRE(curlyVector.x == 12);
		REQUIRE(curlyVector.y == 13);
		REQUIRE(curlyVector.z == 14);
	}
	
	SECTION("setting vector to single value")
	{
		const int randVal = rand();
		myVec = randVal;
		REQUIRE(myVec.x == randVal);
		REQUIRE(myVec.y == randVal);
		REQUIRE(myVec.z == randVal);
	} 

	SECTION("constructor with three elements")
	{
		vector3<int> otherVec(3, 3, 3);
		REQUIRE(otherVec.x == 3);
		REQUIRE(otherVec.y == 3);
		REQUIRE(otherVec.z == 3);
	}

	SECTION("equal operator")
	{
		vector3<int> otherVec(3, 3, 3);
		myVec = 3;
		REQUIRE(myVec == otherVec);
	}

	SECTION("subtraction")
	{
		myVec = {3, 4, 5};
		myVec = myVec - 1;
		REQUIRE(myVec.x == 2);
		REQUIRE(myVec.y == 3);
		REQUIRE(myVec.z == 4);
	}

	SECTION("vector addition")
	{
		vector3<int> otherVec = {1, 2, 3};
		myVec = {1, 0, 4};
		vector3<int> resultVec(2, 2, 7);
		myVec = myVec + otherVec;
		REQUIRE(myVec == resultVec);
		REQUIRE(otherVec.x == 1);
		REQUIRE(otherVec.y == 2);
		REQUIRE(otherVec.z == 3);
	} 

	SECTION("vector multiplication")
	{
		vector3<int> otherVec = {1, 2, 3};
		myVec = {1, 0, 4};
		vector3<int> resultVecMult(1, 0, 12);
		myVec = myVec * otherVec;
		REQUIRE(myVec == resultVecMult);
		REQUIRE(otherVec.x == 1);
		REQUIRE(otherVec.y == 2);
		REQUIRE(otherVec.z == 3);
	} 
	
	SECTION("assignment using curly brackets")
	{
		myVec = {10, 3, 5};
		REQUIRE(myVec.x == 10);
		REQUIRE(myVec.y == 3);
		REQUIRE(myVec.z == 5);
	}

	SECTION("vector access through indexing")
	{
		myVec = {10, 3, 5};
		REQUIRE(myVec[0] == myVec.x);
		REQUIRE(myVec[1] == myVec.y);
		REQUIRE(myVec[2] == myVec.z);
	}

	SECTION("vector assignment")
	{
		vector3<int> otherVec = 3;
		myVec = otherVec;
		REQUIRE(myVec == otherVec);
	}

	SECTION("modulo operator")
	{
		vector3<int> startVal = {5, 6, 7};
		vector3<int> outVal = startVal % 3;
		vector3<int> expectedRes = {2, 0, 1};
		REQUIRE(outVal == expectedRes);
	}
}