#include<iostream>
using namespace std;
int main()
{
    int n;
    cin>>n;
    int m=3*n-2,x=(m-n)/2,y=m-x+1;
    for(int a=0;a<x;a++)
    {
        cout<<" ";
    }
    for(int b=0;b<n;b++)
    {
        cout<<"*";
    }
    cout<<endl;
    for(int c=0;c<n-2;c++)
    {
        for(int d=1;d<=m;d++)
        {
            if(d==x) 
            {
                cout<<"*";
            }
            else if(d==y)
            {
                cout<<"*"<<endl;
                break;
            }
            else
            {
                cout<<" ";
            }
        }
        x--;
        y++;
    }
    for(int e=0;e<3*n-2;e++)
    {
    	cout<<"*";
	}
    return 0;
}
