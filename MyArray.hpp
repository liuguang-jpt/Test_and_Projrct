#pragma once
#include<iostream>
using namespace std;

template<class T>
class MyArray
{
	int m_Capacity;
	int m_Size;
	T* p = NULL;
public:

	MyArray(int capacity)
	{
		this->m_Capacity = capacity;
		this->m_Size = 0;
		this->p = new T[m_Capacity];
	}

	MyArray(const MyArray& arr)
	{
		this->m_Capacity = arr.m_Capacity;
		this->m_Size = arr.m_Size;
		this->p = new T[m_Capacity];
		for (int i = 0; i < this->m_Size; i++)
		{
			this->p[i] = arr.p[i];
		}
	}

	MyArray &operator=(const MyArray& arr)
	{
		if (this == &arr) return *this;

		if (this->p != NULL)
		{
			delete[] this->p;
			this->p = NULL;
		}
		this->m_Capacity = arr.m_Capacity;
		this->m_Size = arr.m_Size;
		
		this->p = new T[m_Capacity];
		for (int i = 0; i < this->m_Size; i++)
		{
			this->p[i] = arr.p[i];
		}
	}

	void MyArrayTailAdd(T temp)
	{
		if (this->m_Size < this->m_Capacity)
		{
			this->p[m_Size] = temp;
			this->m_Size++;
			cout << "增加完毕" << endl;
		}
		else cout << "空间不足" << endl;
	}

	void MyArrayTailDelete()
	{
		if (this->m_Size > 0)
		{
			this->m_Size--;
			cout << "删除完毕" << endl;
		}
		else cout << "内存为空" << endl;
	}

	T& operator[](int i)
	{
		return this->p[i];
	}

	int getCapacity()
	{
		return this->m_Capacity;
	}

	int getSize()
	{
		return this->m_Size;
	}

	~MyArray()
	{
		if (this->p != NULL)
		{
			this->m_Capacity = 0;
			this->m_Size = 0;
			delete[] this->p;
			this->p = NULL;
		}
	}
};