#include<iostream>
using namespace std;
int main()
{
    int m,n,sum=0;
	cin>>m>>n;
	while(m>0&&n>0)
	{
		sum+=m%10+n%10;
		m/=10;
		n/=10;
	}	
	while(1)
	{
		if(sum<10) 
		{
			cout<<sum;
			break;
		}
		else
		{
			int temp=0;
			while(sum>0)
			{
				temp+=sum%10;
				sum/=10;
			}
			sum=temp;
		}
	}
	return 0;
}
