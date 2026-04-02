#include<iostream>
#include<string>
using namespace std;
int main()
{
    string s;
    cin>>s;
    int i=s.size()-1;
    while(s[i]=='9') s[i--]='0';
    if(i>=0) s[i]+=1;
    else s="1"+s;
    cout<<s;
    return 0;
}
