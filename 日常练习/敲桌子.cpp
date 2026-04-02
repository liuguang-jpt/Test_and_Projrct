#include<iostream>
using namespace std;
int main()
{
	for(int a=1;a<=100;a++)
	{
		if(a%10==7)
		{
			cout<<"ÇÃ×À×Ó"<<endl;
		}
		else if(a/10==7)
		{
			cout<<"ÇÃ×À×Ó"<<endl;
		}
		else if(a%7==0)
		{
			cout<<"ÇÃ×À×Ó"<<endl;
	    }
		else
		{
			cout<<a<<endl;
		} 
		
	}
	system("pause");
	return 0;
}
