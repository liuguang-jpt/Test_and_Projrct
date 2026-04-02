#include"Worker.h"
#pragma once
#include<iostream>
using namespace std;
class Employee:public Worker
{
	public:
		Employee(int id,string name,int iId);
		virtual void showInfo();
		virtual string getDeptName();
};
