#include<iostream>
using namespace std;
int f(int n)
{
	int a=1,b=2;
	if(n==1) return a;
	else if(n==2) return b;
	else 
	{
		int current;
		current=f(n-1)+f(n-2);
		return current;
	}
}
int main()
{
	int n;
	cin>>n;
	cout<<f(n)<<endl;
	return 0;
}

