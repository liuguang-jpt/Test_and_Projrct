#include<iostream>
#include"WorkManager.h"
#include"Worker.h"
#include"Employee.h"
#include"Boss.h"
using namespace std;
int main()
{
	WorkManager vm;
	int choice=0;
	while(true)
	{
	    vm.Show_Menu();
	    cout<<"ÇëÊäÈëÄúµÄÑ¡Ôñ£º"<<endl;
	    cin>>choice;
	    switch(choice)
	    {
	    	case 0:
	    		vm.ExitSystem();
	    		break;
	    	case 1:
			    vm.Add_Emp();
				break; 
	    	case 2:
	    		vm.Show_Emp();
	    		break;
	    	case 3:
	    		vm.Del_Emp();
	    		break;
	    	case 4:
	    		vm.Mod_Emp();
	    		break;
	    	case 5:
	    		vm.Find_Emp();
	    		break;
	    	case 6:
	    		vm.Sort_Emp();
	    		break;
	    	case 7:
	    		vm.Clean_Emp();
	    		break;
	    	default:
	    		system("pause");
				system("cls");
		}
    }
	return 0;
}
