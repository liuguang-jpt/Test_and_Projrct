#include<iostream>
using namespace std;
int main()
{
    int n;
    cin>>n;
    for(int m=0;m<2*n-1;m++)
    {
        cout<<"*";
    }
    cout<<endl;
    int m=2*n-2,x=2;
    for(int a=0;a<n-2;a++)
    {
        for(int i=1;i<=2*n-1;i++)
        {
            if(i==x)
            {
                cout<<"*";
            }
            else if(i==m)
            {
                cout<<"*"<<endl;
                break;
            }
            else
            {
                cout<<" ";
            }
        }
        x++;
        m--;
    }
    for(int y=1;y<n;y++)
        {
            cout<<" ";
        }
        cout<<"*";
    return 0;
}
