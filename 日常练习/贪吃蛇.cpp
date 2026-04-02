#include<stdio.h>
#include<graphics.h>
#include<windows.h>
#include<conio.h>
#include<stdlib.h>
#include<math.h>
#include<mmsystem.h>
#pragma comment(lib£¬"Winmm.lib")
#define SNAKE_NUM 500
 
enum DIR
{
	UP,
	DOWN,
	LEFT,
	RIGHT,
};

struct Snake
{
	int size;
	int dir;
	int speed;
	POINT coor[SNAKE_NUM]; 
}snake;

struct Food{


	int x;
	int y;
	int r;
	bool flag;
	DWORD color;
}food;

void Gamelnit()
{
	mciSendString("open ./music/bk.mp3 alias BGM",NULL,0,0);
	mciSendString("play BGM repeat",NULL,0,0);
	MCIERROR result=mciSendString("open old_street.mp3 alias BGM",NULL,0,0);
	if(result !=0)
	{
		char errorMsg[256];
		if(mciGetErrorString(result,errorMsg,sizeof(errorMsg))!=0)
		{
			printf("Failed to open music:Unknown error\n");
		}
	 } 
	 else
	 {
	 	result=mciSendString("play BGM repeat",NULL,0,0);
	 	if(result!=0)
	 	{
	 		char errorMsg[256];
	 		if(mciGetErrorString(result,errorMsg,sizeof(errorMsg))!=0)
	 		{
	 			printf("Failed to play music:Unknown error\n");
			 }
			 else
			 {
			 	printf("Failed to play music:%s\n",errorMsg);
			 }
		 }
	 }
	 initgraph(640,480,SHOWCONSOLE);
	 srand(GetTickCount());
	 snake.size=3;
	 snake.speed=10;
	 snake.dir=RIGHT;
	 for(int i=0;i<snake.size;i++)
	 {
	 	snake.coor[i].x=40-10*i;
	 	snake.coor[i].y=10;
	 	printf("%d %d",snake.coor[i].x,snake.coor[i].y);
	 	food.x=rand()%640;
		food.y=rand()%480;
		food.color=RGB(rand()%256,rand()%256,rand()%256);
		food.r=rand()%10+5;
		food.flag=true;	 }
}
	
	void GameDraw(){
		BeginBatchDraw();
		clearcliprgn();
		setbkcolor(RGB(28,115,119));
		cleardevice();
		setfillcolor(WHITE);
		for(int i=0;i<snake.size;i++)
		{
			solidcircle(snake.coor[i].x,snake.coor[i].y,5);
		}
		if(food.flag)
		{
			solidcircle(food.x,food.y,food.r);
		}
		EndBatchDraw();
    }
	
	void snakeMove()
	{
		for(int i=snake.size-1;i>0;i--)
		{
			snake.coor[i]=snake.coor[i-1];
		}
		switch(snake.dir)
		{
			case UP:
				snake.coor[0].y-=snake.speed;
				if(snake.coor[0].y-5<=0)
				{
					snake.coor[0].y=480;
				}
				break;
			case DOWN:
			    snake.coor[0].y+=snake.speed;
				if(snake.coor[0].y+5>=480)
				{
					snake.coor[0].y=0;
					}
				break;
			case LEFT:
			    snake.coor[0].x-=snake.speed;
				if(snake.coor[0].x-5<=0)
				{
					snake.coor[0].x=640;
				}
				break;
			case RIGHT:
			    snake.coor[0].x+=snake.speed;
				if(snake.coor[0].x+5>=640)
				{
					snake.coor[0].x=0;
				}
				break;
		}
	}
	
	void keyControl()
	{
		if(_kbhit())
		{
			switch(_getch())
			{
			case'w':
			case'W':
			case 72:
				if(snake.dir!=DOWN)
				{
					snake.dir=UP;
				}
				break;
			case's':
			case'S':
			case 80:
				if(snake.dir!=UP)
				{
					snake.dir=DOWN;
				}
				break;
			case'a':
			case'A':
			case 75:
				if(snake.dir!=RIGHT)
				{
					snake.dir=LEFT;
				}
				break;
			case'd':
			case'D':
			case 77:
				if(snake.dir!=LEFT)
				{
					snake.dir=RIGHT;
				}
				break;
			case' ':
				while(1)
				{
					if(_getch()==' ')
					   return;
				}
				break;
			}
		}
	}
	
	void Eatfood(){
	
		if(food.flag && sqrt((snake.coor[0].x-food.x)*
		   (snake.coor[0].x-food.x)+(snake.coor[0].y-food.y)*
		   (snake.coor[0].y-food.y))<=5+food.r)
		{
			food.flag=false;
			snake.size++;
		}
		if(!food.flag)
		{
			food.x=rand()%640;
			food.y=rand()%480;
			food.color=RGB(rand()%256,rand()%256,rand()%256);
			food.r=rand()%10+5;
			food.flag=true;
		}
	}
	
	int main()
	{
		Gamelnit();
		while(1)
		{
			GameDraw();
			snakeMove();
			keyControl();
			Eatfood();
			Sleep(50);
		}
		closegraph();
		return 0;
	}
