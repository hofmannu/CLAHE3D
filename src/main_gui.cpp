/*
  File: main_gui.cpp
  Author: Urs Hofmann
  Mail: mail@hofmannu.org
  Date: 31.03.2022

  Description: simple starter script for our GUI
*/

#include "gui.h"
#include <iostream>

using namespace std;

int main(int argc, char** argv) {
  gui GUI;
  GUI.InitWindow(argc, argv);

  return 0;
}