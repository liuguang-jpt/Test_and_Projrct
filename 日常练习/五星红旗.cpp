#include<bits/stdc++.h>
#include<easyx.h>
#define PI 3.14
void fivestar(int r,double startangle)
{
	double d=2*PI/5;
	POINT p[5];
	    for(int i=0;i<5;i++)
	    {
	    	p[i].x=cos(startangle+i*d*2)*r;
	    	p[i].y=sin(startangle+i*d*2)*r;
		}
		solidpolygon(p,5); 
}
int main(void)
{
	int w=900,h=w/3*2;
	int g=w/30;
	initgraph(w,h);
	setbkcolor(RED);
	cleardevice();
	
	setaspectratio(1,-1);
	setfillcolor(YELLOW);
	setpolyfillmode(WINDING);
	
	setorigin(g*5,g*5);
	fivestar(g*3,PI/2);
	
	double startangle;
	
	setorigin(g*10,g*2);
	startangle=atan(3.0/5.0)+PI;
	fivestar(g,startangle);
	
	setorigin(g*12,g*4);
	startangle=atan(1.0/7.0)+PI;
	fivestar(g,startangle);
	
	setorigin(g*12,g*7);
	startangle=-atan(2.0/7.0)+PI;
	fivestar(g,startangle);
	
	setorigin(g*10,g*9);
	startangle=-atan(4.0/5.0)+PI;
	fivestar(g,startangle);
	getchar();
	closegraph();
	return 0;
}
