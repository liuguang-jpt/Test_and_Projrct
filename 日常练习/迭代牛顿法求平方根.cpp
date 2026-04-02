#include<iostream>
#include<cmath>
#include<stdio.h>
using namespace std;
int main()
{
    double a;
    cin>>a;
    double x=a,y=(x+a/x)/2;
    while(fabs(x-y)>0.00001)
    {
        x=y;
        y=(x+a/x)/2;
    }
    printf("The square root of %.2lf is %lf",a,y);
    return 0;
}
