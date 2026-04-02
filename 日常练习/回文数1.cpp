#include<iostream>
using namespace std;
int main()
{
    int n,k=0,m=0,b=0;
    cin>>n;
    for(int i=1;i<=n;i*=10)//统计位数k 
    {
        if(n/i!=0)
        {
            k++;
        }
    }
    int arr[k];
    int temp=k;
    for(int j=1;j<n;j*=10)
    {
        arr[m++]=n/j%10;//遍历各个位数arr[k] 
    }
    for(int a=0;a<k/2;a++)
    {
        if(arr[a]!=arr[temp-1])//判断回文数1 
        {
            cout<<"No";
        }
        else
        {
            b++;//位数正确记1 
        }
        temp--;
    }
    if(b==k/2)
    {
        cout<<"Yes";//判断回文数2 
    }
    return 0;
}
