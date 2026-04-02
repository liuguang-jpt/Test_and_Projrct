#include <iostream>
using namespace std;
int main() 
{
    int n;
    cin>>n;
    int rows=2*n-1;
    for (int i = 0; i < rows; i++)
    {  
        for (int j = 0; j < rows; j++)
        { 
            if (j == i || j == (2 * n - 2 - i))
            {
                cout << "*";
            } 
            else 
            {
                cout << " ";
            }
        }
        cout << endl; 
    }
    return 0;
}
