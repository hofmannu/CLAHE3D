/*
	File: main_gui.cpp
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 31.03.2022

	Description: simple starter script for our GUI
*/

#include <iostream>
#include "gui.h"

using namespace std;

int main(int *argcp, char**argv)
{
	gui GUI;
	GUI.InitWindow(argcp, argv);

	return 0;
}