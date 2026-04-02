#include<iostream>
#include<vector>
using namespace std;
int main()
{
	int n;
	vector<int>vec;
	cin>>n;
	int arr[n];
	for(int i=0;i<n;i++)
	{
		cin>>arr[i];
	}
	for(int i=0;i<n-1;i++)
	{
		for(int j=0;j<n-1-i;j++)
		{
			if(arr[j]>arr[j+1])
			{
				int temp=arr[j];
				arr[j]=arr[j+1];
				arr[j+1]=temp;
			}
		}
	}
	int count=1;
	for(int i=0;i<n-1;i++)
	{
		if(arr[i+1]=arr[i]==1)
		{
			count++;
		}
		else
		{
			vec.push_back(count);
			count=1;
		}
	}
	for(int i=0;i<vec.size()-1;i++)
	{
		for(int j=0;j<vec.size()-1-i;j++)
		{
			if(vec[j]<vec[j+1])
			{
				int temp=vec[j];
				vec[j]=vec[j+1];
				vec[j+1]=temp;
			}
		}
    }
    cout<<vec[0];
    return 0;
} 
