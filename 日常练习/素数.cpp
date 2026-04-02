#include<iostream>
#include<cmath>
using namespace std;
bool judge (int a)
{
    if(a==1) 
    {
        return false;
    }
    else if(a==2)
    {
        return true;
    }
    else if(a%2==0)
    {
        return false;
    }
    else
    {
        for(int i=3;i<a;i+=2)
        {
            if(a%i==0)
            {
                return false;
            }
        }
    }
    return true;
}
int main()
{
    int m,n;
    cin>>m>>n;
    for(int i=min(m,n);i<=max(m,n);i++)
    {
        if(judge(i))
        {
            cout<<" "<<i;
        }
    }
    return 0;
}
