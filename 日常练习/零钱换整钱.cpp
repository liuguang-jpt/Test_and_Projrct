#include<iostream>
using namespace std;
int main()
{
    int a,b,c;
    for(b=1;b>0;b++)
    {
        for(a=1;a<b;a++)
        {
            for(c=1;c<a;c++)
            {
                if(18*a==15*b&&15*b==20*c)
                {
                    if((a+5*b+10*c)%100==0)
                    {
                        cout<<a<<","<<b<<","<<c<<","<<(a+5*b+10*c)/10<<endl;
                        break;
                    }
                }
            }
            if((a+5*b+10*c)%100==0)
                    {
                        break;
                    }
        }
        if(a+5*b+10*c%100==0)
                    {
                        break;
                    }
    }
    return 0;
}
