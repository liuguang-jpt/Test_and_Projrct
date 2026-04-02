#include<iostream>
using namespace std;

int main()
{
    int n, original, reversed = 0, remainder;
    cin >> n;
    original = n;  // 保存原始数值用于比较
    
    // 反转数字
    while (n > 0)
    {
        remainder = n % 10;
        reversed = reversed * 10 + remainder;
        n = n / 10;
    }
    
    // 比较反转后的数字与原始数字
    if (original == reversed)
        cout << "Yes" << endl;
    else
        cout << "No" << endl;
        
    return 0;
}

