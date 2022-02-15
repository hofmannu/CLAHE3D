#include <iostream>
#include "vector3.h"

int main()
{
	printf("Starting vector3 test.\n");

	srand(1);

	vector3<int> myVec;

	// test initialization with curly brackets
	const vector3<int> curlyVector = {12, 13, 14};
	if ((curlyVector.x != 12) || (curlyVector.y != 13) || (curlyVector.z != 14))
	{
		printf("Curly seems to be unstable\n");
		throw "InvalidResult";
	}
	
	// set all elements in vector to a single value
	const int randVal = rand();
	myVec = randVal;
	if ((myVec.x != randVal) || (myVec.y != randVal) || (myVec.z != randVal))
	{
		printf("Setting the vector did not work\n");
		throw "InvalidResult";
	} 

	// try constructor by passing three elements
	vector3<int> otherVec(3, 3, 3);
	if ((otherVec.x != 3) || (otherVec.y != 3) || (otherVec.z != 3))
	{
		printf("Setting the vector did not work\n");
		throw "InvalidResult";
	}

	// check equal operator
	myVec = 3;
	if (!(myVec == otherVec))
	{
		printf("The two vecotrs should be the same, somethign is wring here\n");
		throw "InvalidValue";
	}

	// check substraction
	myVec = {3, 4, 5};
	myVec = myVec - 1;
	if ((myVec.x != 2) || (myVec.y != 3) || (myVec.z != 4))
	{
		printf("Error while testing substration operation\n");
	}

	// check vector addition
	otherVec = {1, 2, 3};
	myVec = {1, 0, 4};
	vector3<int> resultVec(2, 2, 7);
	myVec = myVec + otherVec;
	if (myVec != resultVec)
	{
		printf("Something went wrong during vector addition.\n");
		throw "InvalidValue";
	}
	if ((otherVec.x != 1) || (otherVec.y != 2) || (otherVec.z !=3))
	{
		printf("Substraction somehow changed our sub vector\n");
		throw "InvalidValue";
	} 

	// check vector multiplication
	otherVec = {1, 2, 3};
	myVec = {1, 0, 4};
	vector3<int> resultVecMult(1, 0, 12);
	myVec = myVec * otherVec;
	if (myVec != resultVecMult)
	{
		printf("Something went wrong during vector multiplication.\n");
		throw "InvalidValue";
	}
	if ((otherVec.x != 1) || (otherVec.y != 2) || (otherVec.z !=3))
	{
		printf("Substraction somehow changed our sub vector\n");
		throw "InvalidValue";
	} 
	
	// check assignemnt using curly brackets
	myVec = {10, 3, 5};
	if ((myVec.x != 10) || (myVec.y != 3) || (myVec.z != 5))
	{
		printf("Setting the vector using curly brackets did not work\n");
		throw "InvalidResult";
	}

	// check vector through indexing
	if ((myVec[0] != myVec.x) || (myVec[1] != myVec.y) || (myVec[2] != myVec.z))
	{
		printf("Access through indexing seems to be not working\n");
		throw "InvalidResult";
	}

	// check vector assignment
	otherVec = 3;
	myVec = otherVec;
	if (myVec != otherVec)
	{
		printf("Something is wrong with vector assignment\n");
		throw "InvalidResult";
	}

	printf("vector3 test executed successfully\n");

	return 0;
};