#include<iostream>
using namespace std;
int main()
{
	int arr[]={3,6,8,1,2,5,7,9,4,0};
	int num=sizeof(arr)/sizeof(arr[0]);
	cout<<"排序前的结果："<<endl;
	for(int c=0;c<num;c++)
	{
		cout<<arr[c];
	}
	cout<<endl;
	cout<<"排序后的结果"<<endl; 
for(int a=0;a<num-1;a++)
{
	for(int i=0;i<num-1;i++)
	{
	if(arr[i]>arr[i+1])//temp在if中定义 
     {
    int temp=arr[i];
	arr[i]=arr[i+1];
	arr[i+1]=temp;
     }
    }
}
    for(int b=0;b<num;b++)
    {
    	cout<<arr[b];
	}
	cout<<endl;
	
	system("pause");
	return 0;
} 
