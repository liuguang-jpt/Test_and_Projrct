#include<iostream>
using namespace std;
int main()
{
    int n;
    cin>>n;
    int x=2*n-1,y=1;
    for(int i=1;i<=n;i++)
    {
        if(i==n)
        {
            for(int j=0;j<2*n-1;j++)
            {
                cout<<"*";
            }
            break;
        }
            for(int b=1;b<=y;b++)
            {
                cout<<"*";
            }
            for(int c=0;c<2*(n-y)-1;c++)
            {
                cout<<" ";
            }
            for(int c=x;c<=2*n-1;c++)
            {
                cout<<"*";
            }
        x--;
        y++;
        cout<<endl;
    }
    return 0;
}
