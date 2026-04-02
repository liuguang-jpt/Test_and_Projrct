#include"manager.h"
Manager::Manager(int id,string name,int dID)
{
	m_ID=id;
	m_Name=name;
	m_DeptID=dID;
}
void Manager::showInfo()
{
	cout<<"职工编号："<<m_ID
	    <<"\t职工姓名："<<m_Name
		<<"\t岗位："<<getDeptName()
		<<"\t岗位职责：完成老板交给的任务，并下发任务给员工"<<endl;
}
string Manager::getDeptName()
{
	return "经理"; 
} 
