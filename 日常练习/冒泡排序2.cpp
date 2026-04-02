#include<iostream>
#include<ctime>
#include<cstdlib>
using namespace std;
int main()
{
	srand(time(NULL));
	cout<<"请输入您想比较的个数，并生成随机数："<<endl;
	int num;
	cin>>num;
	int arr[num];
	for(int a=0;a<num;a++)
	{
		arr[a]=rand()%9999;
	}
	cout<<"排序前的结果："<<endl;
	for(int b=0;b<num;b++)
	{
		cout<<arr[b]<<" ";
	}
	cout<<endl;
	cout<<"排序后的结果"<<endl; 
	for(int c=0;c<num-1;c++)
{
	for(int i=0;i<num-1;i++)
	{
	if(arr[i]>arr[i+1])
     {
    int temp=arr[i];
	arr[i]=arr[i+1];
	arr[i+1]=temp;
     }
    }
}
    for(int d=0;d<num;d++)
    {
    	cout<<arr[d]<<" ";
	}
	cout<<endl;
    
    system("pause");
    return 0;
} 
