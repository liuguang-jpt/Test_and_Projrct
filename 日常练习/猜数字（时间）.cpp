#include<iostream>
using namespace std;
#include<cstdlib>
#include<ctime>
int main()
{
	srand(time(NULL));
	int num=rand()%100+1;
	int val=0;
	while(1)
	{
		cout<<"请输入猜测的数字："<<endl; 
		cin>>val;
		if(val>num)
		{
			cout<<"猜测过大"<<endl;
		} 
		else if(val<num)
		{
			cout<<"猜测过小"<<endl;
		} 
		else 
		{
			cout<<"猜对了"<<endl;
			break;
		}
	} 
	system("pause");
	return 0;
}

