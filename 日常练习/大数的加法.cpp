#include<iostream>
#include<string>
#include<algorithm>
using namespace std;
int main()
{
    string s1,s2,s="";
    cin>>s1>>s2;
    int m=s1.size()-1,n=s2.size()-1,carry=0;
    while(m>=0||n>=0||carry>0)
    {
        int sum=carry;
        if(m>=0) sum+=(int)s1[m--]-'0';
        if(n>=0) sum+=(int)s2[n--]-'0';
        carry=sum/10;
        sum%=10;
        s+=(sum+'0');
    }
    reverse(s.begin(),s.end());
    cout<<s<<endl;
    return 0;
}
