#include<iostream>
using namespace std;
int main()
{
	int n,sum,i; 
	int c=1;
	cin>>n;
	    for(i=1;i<=n;i++)
	    {
		    c=c*i;
		    sum+=c;
	    }
	cout<<sum;
	return 0;
} 
