#include<iostream>
#include<cmath>
using namespace std;
int main()
{
	long long a;
	int i,b,c;
	cin>>a;
	int arr[20];//生成的数组太多不够精细 
	arr[0]=a%10;//个位 
	for(i=1;i<a;i++)
	{
		long long e=pow(10,i+1);
		if(a/e==0)
		{
			c=i+1;//位数 
			break;
		}
	arr[i]=(a*10/e)%10;
    }
    long long f=pow(10,c-1);
    cout<<a/f<<" ";
        for(b=c-2;b>=1;b--)
	    {
		cout<<arr[b]<<" ";
	    }
    cout<<arr[0];
	return 0;
}
