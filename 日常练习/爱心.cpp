#include <iostream>
#include <cmath>
using namespace std;

int main() {
    for (float y = 1.5f; y > -1.5f; y -= 0.1f) {
        for (float x = -1.5f; x < 1.5f; x += 0.05f) {
            // 计算爱心方程
            float a = x * x + y * y - 1;
            if (a * a * a - x * x * y * y * y <= 0.0f) {
                cout << "*";  // 爱心内部用*填充
            } else {
                cout << " ";  // 外部用空格填充
            }
        }
        cout << endl;  // 换行
    }
    return 0;
}
