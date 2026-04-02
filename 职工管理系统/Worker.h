#pragma once
#include<iostream>
#include<string>
using namespace std;
class Worker
{
	public:
		//编号 
		int m_ID;
		//姓名 
		string m_Name;
		//编号 
		int m_DeptID;
		//显示个人信息 
		virtual void showInfo()=0;
		//获取岗位名称 
		virtual string getDeptName()=0;
};
