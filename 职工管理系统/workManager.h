#pragma once
#include<iostream>
#include"Worker.h"
#include"boss.h"
#include"manager.h"
#include"employee.h"
#include<fstream>
#define FILENAME "empFile.txt"
using namespace std;
class WorkManager
{
	public:
		WorkManager();
		
		//展示菜单的函数
		void Show_Menu();
		
		//推出接口 
		void ExitSystem();
		
		//记录职工人数
		int m_EmpNum;
		
		//职工指针数组
		Worker ** m_EmpArray; 
		
		//添加职工
		void Add_Emp(); 
		
		//保存文件
		void save();
		
		//判断文件是否为空
		bool m_FileIsEmpty;
		
		//统计文件中人数
		int get_Empnum();
		
		//初始化职工
		void inti_Emp(); 
		
		//显示职工
		void Show_Emp(); 
		
		//删除职工
		void Del_Emp();
		
		//判断职工是否存在
		int IsExist(int id); 
		
		//修改职工
		void Mod_Emp(); 
		 
		//查找职工
		void Find_Emp(); 
		
		//排序职工
		void Sort_Emp();
		
		//清空数据
		void Clean_Emp();
		 
		~WorkManager();
};
