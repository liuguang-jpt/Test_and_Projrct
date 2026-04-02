#include<iostream>
#include<stdio.h>
#include<cmath>
using namespace std;
int main()
{
    int a,b,c,d,temp;
    scanf("%d/%d+%d/%d",&a,&b,&c,&d);
    for(int i=min(b,d);i>0;i++)
    {
        if(b%i==0&&d%i==0) 
        {
            temp=i;
            break;
        }
    }
    temp=b/temp*d;
    int x=a*temp/b,y=c*temp/d,sum=x+y;
    int m;
    for(int j=min(temp,sum);j>0;j--)
    {
        if(temp%j==0&&sum%j==0)
        {
            m=j;
            break;
        }
    }
    if(m!=1)
    {
        temp/=m;
        sum/=m;
    }
    printf("%d/%d+%d/%d=%d/%d",a,b,c,d,sum,temp);
    return 0;
}
