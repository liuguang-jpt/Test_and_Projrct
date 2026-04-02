#include"boss.h"
Boss::Boss(int id,string name,int dID)
{
	m_ID=id;
	m_Name=name;
	m_DeptID=dID;
}
void Boss::showInfo()
{
	cout<<"职工编号："<<m_ID
	    <<"\t职工姓名："<<m_Name
		<<"\t岗位："<<getDeptName()
		<<"\t岗位职责：管理公司所有事务"<<endl;
}
string Boss::getDeptName()
{
	return "总裁"; 
} 
