#include<iostream>
using namespace std;
int main(){
	int num=100;
	while(num<1000){
		int a=num%10;
		int b=num/10%10;
		int c=num/100;
		if(num==a*a*a+b*b*b+c*c*c){
			cout<<num<<endl;
		}
		num++;
	}
	system("pause");
	return 0;
} 
