#include"workManager.h"

WorkManager::WorkManager()
{
	//文件不存在
	ifstream ifs;
	ifs.open(FILENAME,ios::in);
	
	//文件打开失败，返回false 
	if(!ifs.is_open())
	{
		cout<<"文件不存在"<<endl;
		this->m_EmpNum=0;
        this->m_EmpArray=NULL;
        this->m_FileIsEmpty=true;
        ifs.close();
        return ;
	} 
	
	//文件存在但数据为空
	char ch;
	ifs>>ch;
	if(ifs.eof())
	{
		cout<<"文件为空"<<endl;
		this->m_EmpNum=0;
        this->m_EmpArray=NULL;
        this->m_FileIsEmpty=true;
        ifs.close();
        system("pause");
        system("cls");
	} 
    
    //文件存在，并记录数据
	int num=this->get_Empnum();
	this->m_EmpNum=num;
	//开辟空间 
	this->m_EmpArray=new Worker*[this->m_EmpNum];
	//将文件中的数据，存储到数组中
	this->inti_Emp();
	
}
void WorkManager::Show_Menu()
{
	cout<<"******************************************"<<endl;
	cout<<"******* 欢迎使用职工管理系统 *************"<<endl;
	cout<<"********* 0. 退出管理程序 ****************"<<endl;
	cout<<"********* 1. 增加职工信息 ****************"<<endl;
	cout<<"********* 2. 显示职工信息 ****************"<<endl;
	cout<<"********* 3. 删除离职职工 ****************"<<endl;
	cout<<"********* 4. 修改职工信息 ****************"<<endl;
	cout<<"********* 5. 查找职工信息 ****************"<<endl;
	cout<<"********* 6. 按照编号排序 ****************"<<endl;
	cout<<"********* 7. 清空所有文档 ****************"<<endl;
	cout<<"******************************************"<<endl;
} 
void WorkManager::ExitSystem()
{
	cout<<"欢迎您下次使用！"<<endl;
	system("pause");
	exit(0); 
}
void WorkManager::Add_Emp()
{
	cout<<"请输入添加职工数量："<<endl;
	
	int addNum=0;
	
	cin>>addNum;
	
	if(addNum>0) 
	{
		//计算新空间的大小 
		int NewSize=m_EmpNum+addNum;
		
		//开辟新空间
		Worker **newspace = new Worker*[NewSize];
		
		//将原来空间下的数据，拷贝到新空间下
		if(this->m_EmpArray!=NULL)
		{
			for(int i=0;i<this->m_EmpNum;i++)
			{
				newspace[i]=m_EmpArray[i];
			}
		} 
		
		//添加新数据
		for(int i=0;i<addNum;i++)
		{
			int id;
			string name;
			int dSelect;
			
			cout<<"请输入第"<<i+1<<"个新职工的编号："<<endl;
			cin>>id;
			cout<<"请输入第"<<i+1<<"个新职工的姓名："<<endl;
			cin>>name; 
			cout<<"请选择第该职工的岗位："<<endl;
			cout<<"1、普通职工"<<endl;
			cout<<"2、经理"<<endl;
			cout<<"3、老板"<<endl; 
			cin>>dSelect;
			
			Worker *worker=NULL;
			switch(dSelect)
			{
				case 1:
					worker=new Employee(id,name,1);
					break;
				case 2:
					worker=new Manager(id,name,2);
					break;
				case 3:
					worker=new Boss(id,name,3);
					break;
				default:
					break;
			} 
			//将创建的职工责任，保存到数组中 
			newspace[this->m_EmpNum+i]=worker;
			
		} 
		
		//释放原有空间
		delete [] m_EmpArray;
		
		this->m_EmpArray=newspace; 
		this->m_EmpNum=NewSize;
		
		this->m_FileIsEmpty=false;
		
		//提示添加成功
		cout<<"成功添加"<<addNum<<"名新职工"<<endl;
		 
		//保存数据到文件中 
		this->save();
	}
	else 
	{
		cout<<"输入数据有误"<<endl;
	}
	
	//按任意键后，清屏回到上级目录
	system("pause");
	system("cls"); 
}

//保存文件
void WorkManager::save()
{
	ofstream ofs;
	ofs.open(FILENAME,ios::out);
	
	for(int i=0;i<this->m_EmpNum;i++)
	{
		ofs<<this->m_EmpArray[i]->m_ID<<" "
		<<this->m_EmpArray[i]->m_Name<<" "
		<<this->m_EmpArray[i]->m_DeptID<<endl;
	}
	ofs.close();
} 

//统计文件中的人数
int WorkManager::get_Empnum()
{
	ifstream ifs;
	ifs.open(FILENAME,ios::in);
	
	int id;
	string name;
	int did;
	
	int num=0;
	
	while(ifs>>id&&ifs>>name&&ifs>>did)
	{
		num++;
	}
	return num; 
}

//初始化员工
void WorkManager::inti_Emp()
{
	ifstream ifs;
	ifs.open(FILENAME,ios::in);
	
	int id;
	string name;
	int did;
	
	int index=0;
	while(ifs>>id&&ifs>>name&&ifs>>did)
	{
		Worker *worker=NULL;
		if(did==1) worker=new Employee(id,name,did);
		else if(did==2) worker=new Manager(id,name,did);
		else worker=new Boss(id,name,did);
		this->m_EmpArray[index++]=worker;
	}
	
	ifs.close(); 
} 

void WorkManager::Show_Emp()
{
	//判断文件是否为空 
	if(this->m_FileIsEmpty)
	{
		cout<<"文件不存在或记录为空!"<<endl;
	}
	else 
	{
		for(int i=0;i<this->m_EmpNum;i++)
		{
			//利用多态调用程序接口
			this->m_EmpArray[i]->showInfo(); 
		}
	}
	
	system("pause");
	system("cls");
}

//删除职工 
void WorkManager::Del_Emp()
{
	if(this->m_FileIsEmpty)
	{
		cout<<"文件不存在或记录为空！"<<endl;
	}
	else 
	{
		cout<<"请输入想删除的职工编号："<<endl;
		int id=0;
		cin>>id;
		
		int index=this->IsExist(id);
		
		if(index!=-1)
		{
			//数据迁移
			for(int i=index;i<this->m_EmpNum-1;i++)
			{
				this->m_EmpArray[i]=this->m_EmpArray[i+1];
			} 
			this->m_EmpNum--;
			//数据保存在文件中
			this->save();
			
			cout<<"删除成功！"<<endl; 
		} 
		else 
		{
			cout<<"删除失败，未找到该职工"<<endl; 
		}
		
		system("pause");
		system("cls");
	}
}

//判断职工是否存在  若存在返回职工在数组中的位置  不存在返回-1 
int WorkManager::IsExist(int id)
{
	int index=-1;
	for(int i=0;i<this->m_EmpNum;i++)  
	{
		if(this->m_EmpArray[i]->m_ID==id) 
		{
			index=i;
			break;
		}
	}
	return index;
}

//修改职工
void WorkManager::Mod_Emp()
{
	if(this->m_FileIsEmpty)
	{
		cout<<"文件不存在或记录为空！"<<endl;
	}
	else 
	{
		cout<<"请输入修改职工编号："<<endl;
		int id;
		cin>>id;
		
		int ret=this->IsExist(id);
		if(ret!=-1)
		{
			delete this->m_EmpArray[ret];
			
			int newID=0;
			string newName="";
			int dSelect=0;
		    cout<<"查找到"<<id<<"号职工，请输入新职工号"<<endl;
		    cin>>newID;
		
		    cout<<"请输入新姓名："<<endl;
	    	cin>>newName;
		
	    	cout<<"请输入岗位："<<endl;
	    	cout<<"1、普通员工"<<endl;
	    	cout<<"2、经理"<<endl;
	    	cout<<"3、老板"<<endl;
	    	cin>>dSelect;
		 
	    	Worker *worker=NULL;
	    	switch(dSelect)
	    	{
		    	case 1:
			    	worker=new Employee(newID,newName,dSelect);
			    	break;
		    	case 2:
			    	worker=new Manager(newID,newName,dSelect);
			    	break;
		    	case 3:
			    	worker=new Boss(newID,newName,dSelect);
			    	break;
			    default:
			    	break;
	    	}
		
	    	//更新数据到数组中
	    	this->m_EmpArray[ret]=worker;
	    	cout<<"修改成功！"<<endl;
		
	    	//保存文件中
	    	this->save();
    	}
	    else 
		{
		 	cout<<"修改失败，查无此人！"<<endl;
		}
		
		system("pause");
		system("cls");
	}
}

//查找职工
void WorkManager::Find_Emp()
{
	if(this->m_FileIsEmpty)
	{
		cout<<"文件不存在或记录为空！"<<endl; 
	}
	else 
	{
		cout<<"请输入查找方式"<<endl;
		cout<<"1、按照职工编号查找"<<endl;
		cout<<"2、按照职工姓名查找"<<endl;
		
		int select=0;
		cin>>select;
		
		if(select==1)
		{
			int id;
			cout<<"请输入查找的职工编号："<<endl;
			cin>>id;
			
			int ret=IsExist(id);
			if(ret!=-1)
			{
				cout<<"查找成功！该职工信息如下："<<endl;
				this->m_EmpArray[ret]->showInfo();
			}
			else 
			{
				cout<<"查找失败！查无此人"<<endl;
			}
		}
		else if(select==2)
		{
			string name;
			cout<<"请输入查找职工的姓名："<<endl;
			cin>>name;
			
			//加入判断是否查找到的标志
			bool flag=false; 
			
			for(int i=0;i<this->m_EmpNum;i++)
			{
				if(this->m_EmpArray[i]->m_Name==name)
				{
					cout<<"查找成功，职工编号为："<<this->m_EmpArray[i]->m_ID
					    <<"号职工的信息如下："<<endl;
					    
					this->m_EmpArray[i]->showInfo();
					
					flag=true;
				}
			}
			if(!flag)
			{
				cout<<"查找失败，查无此人"<<endl;
			}
		}
		else 
		{
			cout<<"输入选项有误！"<<endl;
		}
		
		system("pause");
		system("cls");
	}
} 

//按编号排序 
void WorkManager::Sort_Emp()
{
	if(this->m_FileIsEmpty)
	{
		cout<<"文件不存在或记录为空！"<<endl;
		system("pause");
		system("cls");
	}
	else 
	{
		cout<<"请选择排序方式："<<endl;
		cout<<"1、按职工号进行升序"<<endl;
		cout<<"2、按职工号进行降序"<<endl; 
	}
	int select=0;
	cin>>select;
	
	for(int i=0;i<this->m_EmpNum;i++)
    	{
		    int minOrmax=i;
		    for(int j=i+1;j<this->m_EmpNum;j++)
		    {
		        if(select==1)
				{
					//升序 
					if(this->m_EmpArray[minOrmax]->m_ID > this->m_EmpArray[j]->m_ID) minOrmax=j;
				} 
			    else 
			    {
			    	//降序
					if(this->m_EmpArray[minOrmax]->m_ID < this->m_EmpArray[j]->m_ID) minOrmax=j;
				}
	    	}
	    	if(i!=minOrmax)
			{
				Worker *temp=this->m_EmpArray[i];
				this->m_EmpArray[i]=this->m_EmpArray[minOrmax];
				this->m_EmpArray[minOrmax]=temp;
			} 
	    }
	    
	    cout<<"排序成功！排序后的结果为："<<endl;
	    this->save();
	    this->Show_Emp();
}

//清空文件
void WorkManager::Clean_Emp()
{
	cout<<"确定清空："<<endl;
	cout<<"1、确定"<<endl;
	cout<<"2、返回"<<endl;
	
	int select=0;
	cin>>select; 
	if(select==1)
	{
		ofstream ofs(FILENAME, ios::trunc);
		ofs.close();
		
		if(this->m_EmpArray!=NULL)
		{
			for(int i=0;i<this->m_EmpNum;i++)
			{
				delete this->m_EmpArray[i];
				this->m_EmpArray[i]=NULL;
			}
			
			delete[] this->m_EmpArray;
			this->m_EmpArray=NULL;
			this->m_EmpNum=0;
			this->m_FileIsEmpty=true;
			
			cout<<"清空成功！"<<endl;
		}
		
		system("pause");
		system("cls");
	}
	else
	{
		system("pause");
		system("cls");
	}
}
 
WorkManager::~WorkManager()
{
    if(this->m_EmpArray!=NULL)
    {
    	for(int i=0;i<this->m_EmpNum;i++)
    	{
    		if(this->m_EmpArray[i]!=NULL)
    		{
    			delete this->m_EmpArray[i];
			}
		}
		
		delete []this->m_EmpArray;
		this->m_EmpArray=NULL;
	}
}
