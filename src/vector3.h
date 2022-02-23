/* 
	This is the most useless class ever, but a good training. Handles 3 element vectors
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 15.02.2022
*/

#ifndef VECTOR3_H
#define VECTOR3_H

#include <cstdio>
#include <cstdint>

template<typename T>
class vector3
{

public:
	T x;
	T y;
	T z;

	// class constructors
	vector3()
	{
		x = 0; y = 0; z = 0;
	}

	vector3(const T initVal)
	{
		x = initVal; y = initVal; z = initVal;
	}

	vector3(const T xInit, const T yInit, const T zInit)
	{
		x = xInit; y = yInit; z = zInit;
	}

	T operator[] (const std::size_t idx) const
	{
		return *((&x) + idx);
	}

	// check if two vectors are the same
	bool operator == (const vector3& B) const
	{
		return ((x == B.x) && (y == B.y) && (z * B.z));
	}

	// check if two vectors are different
	bool operator != (const vector3& B) const 
	{
		return ((x != B.x) || (y != B.y) || (z != B.z));
	}

	// assign same value to all three coordinates
	void operator = (const T setVal) {
		x = setVal;	y = setVal;	z = setVal;
	}

	// assign a vector to a vector
	vector3& operator = (const vector3& setVal) {
		x = setVal.x;	y = setVal.y;	z = setVal.z;
		return *this;
	}	

	// basic math with constants
	vector3 operator * (const T multVal)
	{
		vector3 output = {x * multVal,	y * multVal,	z * multVal};
		return output;
	}

	vector3 operator / (const T divVal)
	{
		vector3 output = {x / divVal,	y / divVal,	z / divVal};
		return output;
	}

	// substraction
	vector3 operator - (const T subVal) const
	{
		vector3 output = {x - subVal,	y - subVal,	z - subVal};
		return output;
	}

	vector3 operator + (const T addVal) const
	{
		vector3 output = {x + addVal, y + addVal, z + addVal};
		return output;
	}

	// basic elementwise math with other vector
	vector3 operator * (const vector3& multVec)
	{
		vector3 output = {x * multVec.x,	y * multVec.y,	z * multVec.z};
		return output;
	}

	// basic elementwise math with other vector
	vector3 operator * (const vector3& multVec) const
	{
		const vector3 output = {x * multVec.x, y * multVec.y,	z * multVec.z};
		return output;
	}

	vector3 operator / (const vector3& divVec)
	{
		const vector3 output = {x / divVec.x,	y / divVec.y,	z / divVec.z};
		return output;
	}

	vector3 operator - (const vector3& subVec) const
	{
		vector3 output = {x - subVec.x,	y - subVec.y,	z - subVec.z};
		return output;
	}

	vector3 operator + (const vector3& addVec) const
	{
		vector3 output = {x + addVec.x,	y + addVec.y,	z + addVec.z};
		return output;
	}

	// returns true if any element has that value
	bool any(const T compValue) const
	{
		if ((x == compValue) || (y == compValue) || (z == compValue))
			return 1;
		else
			return 0;
	}

	inline T elementSum() const
	{
		return (x + y + z);
	}

	inline T elementMult() const
	{
		return (x * y * z);
	}

};



#endif