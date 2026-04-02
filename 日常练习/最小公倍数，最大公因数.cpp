#include<iostream>
#include<cmath>
using namespace std;
int main()
{
	int m,n,d,x;
	cin>>m>>n;
	for(int a=min(m,n);a>0;a--)
	{
		if(m%a==0&&n%a==0)
		{
			d=a;
			break;
		}
	}
	x=m/d*n;
	cout<<"the greatest common divisor is "<<d<<endl<<"the least common multiple is "<<x<<endl;
	return 0;
}
