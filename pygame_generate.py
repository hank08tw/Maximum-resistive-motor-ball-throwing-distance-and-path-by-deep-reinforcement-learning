output_path='./data3.txt'
import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
#from gym.envs.toy_text import discretepd.read_json("tzzs_data.json")

from collections import defaultdict
#from cliff_walking import CliffWalkingEnv
import plotting

import math
from math import sin as sin
from math import cos as cos
from math import floor as floor
#from tkinter import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
import svgwrite
from random import randint
import numpy as np
import time
import sys
# 14965 39658 41758
import math
from math import sin as sin
from math import cos as cos
from math import floor as floor
import gym
import matplotlib
import plotting
import cv2
import svgwrite
from random import randint
import matplotlib.pyplot as plt

from PIL import Image
from pylab import *
matplotlib.style.use('ggplot')
import numpy as np
import pandas as pd
#import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

matplotlib.style.use('ggplot')

#最後svg動畫和theta vs w的圖會畫這個episode的結果
print_this_episode_number=69
count=0
#上面能擺動的最大力矩
L_m1=1.3
#中間電機能擺動的最大力矩
L_m2=1.2
#現在這個state的上面電機提供力矩大小 可以是L_m,0,-L_m
L_1=0
#現在這個state的中間電機提供力矩大小
L_2=0
#上面木棒的角度（是弧度）
theta_1=0
#中間木棒的角度
theta_2=0
#現在的角速度
w_1=0
w_2=0
#現在的角加速度
alpha_1=0
alpha_2=0
#重力大小
g=9.98
#模型的時間間隔
delta_t=0.01#0.001
#總時間
total_time=0.0
#最大時間
max_total_time=50.0
#畫圖的時間間隔
draw_delta_t=0.01
#木棒的長度
R_1=0.5
R_2=0.5
#畫圖的木棒長度
draw_R_1=50
draw_R_2=50
#球的質量
m_1=0.5
m_2=0.5
#阻力係數k*w
k=0.01#0.05
#電機沒有力矩的阻力係數（目前還沒用）
k2=0.005
#畫圖的時間軸
i=0
#目前找到最大路徑是哪個episode
max_x_num=-1
#最大路徑episode丟出的距離
max_x_val=-2147483647
#即時的reward，此係數乘上總能量當作reward
energy_coefficient=1
#最後的reward，此係數乘上拋出的距離當作最後的reward
last_coefficient=1
#即時的reward，此係數呈上能量差當作reward
energy_diff_coefficient=1#0.1 1.16 1 421
#算法跑幾個episode
num_episode=10000
# reproducible
np.random.seed(1)
#tf.set_random_seed(1)

#用來畫圖 儲存每個時間點theta跟w的值
theta_box_1 = []
w_box_1 = []
alpha_box_1= []
moment_1=[]

theta_box_2 = []
w_box_2 = []
alpha_box_2 = []
moment_2 = []

x_box_1=[]
y_box_1=[]

x_box_2=[]
y_box_2=[]
"""
in_circle_line_1=image.add(image.circle((100+draw_R_1+draw_R_2,0),1))
in_circle_line_2=image.add(image.circle((100+draw_R_1+draw_R_2,0),1))
in_circle_line_3=image.add(image.circle((100+draw_R_1+draw_R_2,0),1))
in_circle_line_4=image.add(image.circle((100+draw_R_1+draw_R_2,0),1))
in_circle_line_5=image.add(image.circle((100+draw_R_1+draw_R_2,0),1))
in_circle_line_6=image.add(image.circle((100+draw_R_1+draw_R_2,0),1))
in_circle_line_7=image.add(image.circle((100+draw_R_1+draw_R_2,0),1))
in_circle_line_8=image.add(image.circle((100+draw_R_1+draw_R_2,0),1))
in_circle_line_9=image.add(image.circle((100+draw_R_1+draw_R_2,0),1))
in_circle_line_10=image.add(image.circle((100+draw_R_1+draw_R_2,0),1))

out_circle_line_1=image.add(image.circle((100+draw_R_1+draw_R_2,0),1))
out_circle_line_2=image.add(image.circle((100+draw_R_1+draw_R_2,0),1))
out_circle_line_3=image.add(image.circle((100+draw_R_1+draw_R_2,0),1))
out_circle_line_4=image.add(image.circle((100+draw_R_1+draw_R_2,0),1))
out_circle_line_5=image.add(image.circle((100+draw_R_1+draw_R_2,0),1))
out_circle_line_6=image.add(image.circle((100+draw_R_1+draw_R_2,0),1))
out_circle_line_7=image.add(image.circle((100+draw_R_1+draw_R_2,0),1))
out_circle_line_8=image.add(image.circle((100+draw_R_1+draw_R_2,0),1))
out_circle_line_9=image.add(image.circle((100+draw_R_1+draw_R_2,0),1))
out_circle_line_10=image.add(image.circle((100+draw_R_1+draw_R_2,0),1))
"""


image = svgwrite.Drawing('strategy_2.svg', size=(1024,1024))

in_circle_line_num=10
in_circle_line=[]
out_circle_line_num=10
out_circle_line=[]

for i in range(in_circle_line_num):
    in_circle_line.append(image.add(image.circle((100+draw_R_1+draw_R_2,0),1)))
    
for i in range(out_circle_line_num):
    out_circle_line.append(image.add(image.circle((100+draw_R_1+draw_R_2,0),1)))


action_space=['-L_m,-L_m','-L_m,0','-L_m,L_m','0,-L_m','0,0','0,L_m','L_m,-L_m','L_m,0','L_m,L_m','shoot']

mid=image.add(image.circle((100+draw_R_1+draw_R_2,0),5))
circle=image.add(image.circle((100+draw_R_1+draw_R_2,0),5))
circle2=image.add(image.circle((100+draw_R_1+draw_R_2,0),5))
#line_1=image.add(image.line((100+draw_R_1+draw_R_2, 0), (100+draw_R_1+draw_R_2, -draw_R_1), stroke=svgwrite.rgb(0, 0, 0, '%')))
#line_2=image.add(image.line((100+draw_R_1+draw_R_2, -draw_R_1), (100+draw_R_1+draw_R_2, -draw_R_1-draw_R_2), stroke=svgwrite.rgb(0, 0, 0, '%')))

image.add(image.line((0, 400-(draw_R_1+draw_R_2)), (1024, 400-(draw_R_1+draw_R_2)), stroke=svgwrite.rgb(0, 0, 0, '%')))
#image.add()




def shoot_the_ball2(x,y,v,theta,print_this_episode):
    global g,delta_t,i
    time=(v*math.sin(theta)+math.sqrt(v*v*math.sin(theta)*math.sin(theta)+2*g*y))/g
    if print_this_episode:
        tmp_t=0
        while tmp_t<=math.floor(time) and x+v*math.cos(theta)*tmp_t<1000:
            s_old=y+v*math.sin(theta)*tmp_t-0.5*g*tmp_t*tmp_t
            s_new=y+v*math.sin(theta)*(tmp_t+delta_t)-0.5*g*(tmp_t+delta_t)*(tmp_t+delta_t)
            #path = 'M' + str(x+v*math.cos(theta)*tmp_t) + ',' + str(-s_old+400) + ' L ' + str(x+v*math.cos(theta)*(tmp_t+delta_t)) + ',' + str(-s_new+400)
            #circle.add(image.animateMotion(path=path,dur=str(draw_delta_t)+"s", begin=str(i*draw_delta_t)+"s",fill="freeze"))
            #i=i+1
            tmp_t=tmp_t+delta_t
    return v*math.cos(theta)*time
#%matplotlib inline

observation_array=[]
reward_array=[]
def do_action(action,print_this_episode):
    #print('math.pi: '+str(math.pi))
    global L_m1,L_m2,L_1,L_2,theta_1,theta_2,w_1,w_2,alpha_1,alpha_2,g,delta_t,R_1,R_2,m_1,m_2,k,k2,i,max_x_num,max_x_val,action_space,count,theta_box_1,theta_box_2,w_box_1,w_box_2,alpha_box_1,alpha_box_2,energy_coefficient,last_coefficient,energy_diff_coefficient,total_time,max_total_time,in_circle_line,in_circle_line_num,out_circle_line,out_circle_line_num,observation_array
    action_array.append(action)
    observation_array.append([sin(theta_1),cos(theta_1),w_1,sin(theta_2),cos(theta_2),w_2])
    prev_x_in=R_1*sin(theta_1)
    prev_y_in=R_1*cos(theta_1)
    prev_x_out=R_2*sin(theta_2)
    prev_y_out=R_2*cos(theta_2)
    prev_x=R_1+R_2+R_1*sin(theta_1)+R_2*sin(theta_2)
    #print('==================theta_1:'+str(theta_1))
    prev_y=R_1+R_2-R_1*cos(theta_1)-R_2*cos(theta_2)
    #print('theta_2:'+str(theta_2))
    v_1=R_1*w_1
    v_2=R_2*w_2
    v_3=math.sqrt(v_1*v_1+v_2*v_2-2*v_1*v_2*cos(math.pi-(theta_1-theta_2)))
    #print('v3:'+str(v_3))
    prev_energy=0.5*m_1*v_1*v_1+0.5*m_2*v_3*v_3+m_1*g*(R_1+R_2-R_1*cos(theta_1))+m_2*g*(R_1+R_2-R_1*cos(theta_1)-R_2*cos(theta_2))
    total_time+=delta_t
    if total_time>max_total_time:
        print('time out!')
        reward_array.append(0)
        return np.array([sin(theta_1),cos(theta_1),w_1,sin(theta_2),cos(theta_2),w_2]),0,True
    elif action==9:
        tmp=shoot_the_ball2(prev_x,prev_y,v_3,theta_2,print_this_episode)+R_1*sin(theta_1)+R_2*math.sin(theta_2)
        print('distance: '+str(tmp))
        if tmp>max_x_val:
            max_x_val=tmp
            max_x_num=count
        reward_array.append(tmp*last_coefficient)
        return np.array([sin(theta_1),cos(theta_1),w_1,sin(theta_2),cos(theta_2),w_2]),tmp*last_coefficient,True
    elif action==0:
        L_1=-L_m1
        L_2=-L_m2
        #print(m_1*L_1*L_1+m_2*abs(L_1*L_1+L_2*L_2-2*L_1*L_2*cos(math.pi-(theta_1-theta_2))))
        alpha_1=(L_1-m_1*g*sin(theta_1)*R_1-m_2*g*(R_1*sin(theta_1)+R_2*sin(theta_2)))/(m_1*R_1*R_1+m_2*(R_1*R_1+R_2*R_2-2*R_1*R_2*cos(math.pi-(theta_1-theta_2))))
        alpha_2=(L_2-m_2*g*R_2*sin(theta_2))/(m_2*R_2*R_2)+alpha_1
        w_1=w_1+alpha_1*delta_t-k*w_1
        w_2=w_2+alpha_2*delta_t-k*w_2
    elif action==1:
        L_1=-L_m1
        L_2=0
        #print(m_1*L_1*L_1+m_2*abs(L_1*L_1+L_2*L_2-2*L_1*L_2*cos(math.pi-(theta_1-theta_2))))
        alpha_1=(L_1-m_1*g*sin(theta_1)*R_1-m_2*g*(R_1*sin(theta_1)+R_2*sin(theta_2)))/(m_1*R_1*R_1+m_2*(R_1*R_1+R_2*R_2-2*R_1*R_2*cos(math.pi-(theta_1-theta_2))))
        alpha_2=(L_2-m_2*g*R_2*sin(theta_2))/(m_2*R_2*R_2)+alpha_1
        w_1=w_1+alpha_1*delta_t-k*w_1
        w_2=w_2+alpha_2*delta_t-k2*w_2
    elif action==2:
        L_1=-L_m1
        L_2=L_m2
        #print(m_1*L_1*L_1+m_2*abs(L_1*L_1+L_2*L_2-2*L_1*L_2*cos(math.pi-(theta_1-theta_2))))
        alpha_1=(L_1-m_1*g*sin(theta_1)*R_1-m_2*g*(R_1*sin(theta_1)+R_2*sin(theta_2)))/(m_1*R_1*R_1+m_2*(R_1*R_1+R_2*R_2-2*R_1*R_2*cos(math.pi-(theta_1-theta_2))))
        alpha_2=(L_2-m_2*g*R_2*sin(theta_2))/(m_2*R_2*R_2)+alpha_1
        w_1=w_1+alpha_1*delta_t-k*w_1
        w_2=w_2+alpha_2*delta_t-k*w_2
    elif action==3:
        L_1=0
        L_2=-L_m2
        #print(m_1*L_1*L_1+m_2*abs(L_1*L_1+L_2*L_2-2*L_1*L_2*cos(math.pi-(theta_1-theta_2))))
        alpha_1=(L_1-m_1*g*sin(theta_1)*R_1-m_2*g*(R_1*sin(theta_1)+R_2*sin(theta_2)))/(m_1*L_1*L_1+m_2*(R_1*R_1+R_2*R_2-2*R_1*R_2*cos(math.pi-(theta_1-theta_2))))
        alpha_2=(L_2-m_2*g*R_2*sin(theta_2))/(m_2*R_2*R_2)+alpha_1
        w_1=w_1+alpha_1*delta_t-k2*w_1
        w_2=w_2+alpha_2*delta_t-k*w_2
    elif action==4:
        L_1=0
        L_2=0
        alpha_1=(L_1-m_1*g*sin(theta_1)*R_1-m_2*g*(R_1*sin(theta_1)+R_2*sin(theta_2)))/(m_1*L_1*L_1+m_2*(R_1*R_1+R_2*R_2-2*R_1*R_2*cos(math.pi-(theta_1-theta_2))))
        alpha_2=(L_2-m_2*g*R_2*sin(theta_2))/(m_2*R_2*R_2)+alpha_1
        w_1=w_1+alpha_1*delta_t-k2*w_1
        w_2=w_2+alpha_2*delta_t-k2*w_2
    elif action==5:
        L_1=0
        L_2=L_m2
        #print(m_1*L_1*L_1+m_2*abs(L_1*L_1+L_2*L_2-2*L_1*L_2*cos(math.pi-(theta_1-theta_2))))
        alpha_1=(L_1-m_1*g*sin(theta_1)*R_1-m_2*g*(R_1*sin(theta_1)+R_2*sin(theta_2)))/(m_1*L_1*L_1+m_2*(R_1*R_1+R_2*R_2-2*R_1*R_2*cos(math.pi-(theta_1-theta_2))))
        alpha_2=(L_2-m_2*g*R_2*sin(theta_2))/(m_2*R_2*R_2)+alpha_1
        w_1=w_1+alpha_1*delta_t-k2*w_1
        w_2=w_2+alpha_2*delta_t-k*w_2
    elif action==6:
        L_1=L_m1
        L_2=-L_m2
        #print(m_1*L_1*L_1+m_2*abs(L_1*L_1+L_2*L_2-2*L_1*L_2*cos(math.pi-(theta_1-theta_2))))
        alpha_1=(L_1-m_1*g*sin(theta_1)*R_1-m_2*g*(R_1*sin(theta_1)+R_2*sin(theta_2)))/(m_1*L_1*L_1+m_2*(R_1*R_1+R_2*R_2-2*R_1*R_2*cos(math.pi-(theta_1-theta_2))))
        alpha_2=(L_2-m_2*g*R_2*sin(theta_2))/(m_2*R_2*R_2)+alpha_1
        w_1=w_1+alpha_1*delta_t-k*w_1
        w_2=w_2+alpha_2*delta_t-k*w_2
    elif action==7:
        L_1=L_m1
        L_2=0
        #print(m_1*L_1*L_1+m_2*abs(L_1*L_1+L_2*L_2-2*L_1*L_2*cos(math.pi-(theta_1-theta_2))))
        alpha_1=(L_1-m_1*g*sin(theta_1)*R_1-m_2*g*(R_1*sin(theta_1)+R_2*sin(theta_2)))/(m_1*L_1*L_1+m_2*(R_1*R_1+R_2*R_2-2*R_1*R_2*cos(math.pi-(theta_1-theta_2))))
        alpha_2=(L_2-m_2*g*R_2*sin(theta_2))/(m_2*R_2*R_2)+alpha_1
        w_1=w_1+alpha_1*delta_t-k*w_1
        w_2=w_2+alpha_2*delta_t-k2*w_2
    elif action==8:
        L_1=L_m1
        L_2=L_m2
        #print(m_1*L_1*L_1+m_2*abs(L_1*L_1+L_2*L_2-2*L_1*L_2*cos(math.pi-(theta_1-theta_2))))
        alpha_1=(L_1-m_1*g*sin(theta_1)*R_1-m_2*g*(R_1*sin(theta_1)+R_2*sin(theta_2)))/(m_1*L_1*L_1+m_2*(R_1*R_1+R_2*R_2-2*R_1*R_2*cos(math.pi-(theta_1-theta_2))))
        alpha_2=(L_2-m_2*g*R_2*sin(theta_2))/(m_2*R_2*R_2)+alpha_1
        w_1=w_1+alpha_1*delta_t-k*w_1
        w_2=w_2+alpha_2*delta_t-k*w_2
    else:
        print("action not exist!!!")
    #w=w+(L-R*math.sin(theta)*m*g)/(m*R*R)-k*w
    
    theta_1=w_1*delta_t+theta_1
    theta_2=w_2*delta_t+theta_2
    cur_x_in=R_1*sin(theta_1)
    cur_y_in=R_1*cos(theta_1)
    cur_x_out=R_2*sin(theta_2)
    cur_y_out=R_2*cos(theta_2)
    cur_x=R_1+R_2+R_1*sin(theta_1)+R_2*sin(theta_2)
    cur_y=R_1+R_2-R_1*cos(theta_1)-R_2*cos(theta_2)
    v_1=R_1*w_1
    v_2=R_2*w_2
    v_3=math.sqrt(v_1*v_1+v_2*v_2-2*v_1*v_2*cos(math.pi-(theta_1-theta_2)))
    energy=0.5*m_1*v_1*v_1+0.5*m_2*v_3*v_3+m_1*g*(R_1+R_2-R_1*cos(theta_1))+m_2*g*(R_1+R_2-R_1*cos(theta_1)-R_2*cos(theta_2))
    energy_diff=energy-prev_energy

    reward_array.append(energy_diff_coefficient*energy_diff)    
    return np.array([math.sin(theta_1),math.cos(theta_1),w_1,sin(theta_2),cos(theta_2),w_2]),energy_diff_coefficient*energy_diff,False

import numpy as np
#import tensorflow as tf

action_array=[]




#打飛船射級遊戲
import pygame,random,math,time
#玩家操控的太空船
"""
class Enemy(pygame.sprite.Sprite):
    speed=0
    def __init__(self,color,x,y,speed):
        pygame.sprite.Sprite.__init__(self)
        self.image=pygame.Surface([25,25])
        self.image.fill(color)
        self.rect=self.image.get_rect()
        self.rect.x=x
        self.rect.y=y
        self.speed=speed
    def update(self):
        self.rect.y+=self.speed
class Bullet(pygame.sprite.Sprite):
    def __init__(self,color,x,y,speed):
        pygame.sprite.Sprite.__init__(self)
        self.image=pygame.Surface([20,20])
        self.image.fill((255,255,255))
        pygame.draw.circle(self.image,(color),(10,10),5,0)
        self.rect=self.image.get_rect()
        self.rect.x=x
        self.rect.y=y
        self.speed=speed
    def update(self):
        self.rect.y-=self.speed
"""
class theObject(pygame.sprite.Sprite):
    def __init__(self,color,x,y):
        pygame.sprite.Sprite.__init__(self)
        self.image=pygame.Surface([10,10])
        self.image.fill(color)
        self.rect=self.image.get_rect()#取得這個image的位置
        self.rect.x=x
        self.rect.y=y
    def update(self,x,y):
        self.rect.x=x
        self.rect.y=y
    """
    def update(self):
        #取得滑鼠的位置
        pos=pygame.mouse.get_pos()
        #把太空船的位置設成滑鼠左鍵按下的位置
        self.rect.x=pos[0]
        #不能越過邊界
        if self.rect.x>window.get_width()-self.rect.width:
            self.rect.x=window.get_width()-self.rect.width
        elif self.rect.x<0:
            self.rect.x=0
    """

pygame.init()
font=pygame.font.SysFont("SimHei",100)
def gameover(message):
    global run
    text=font.render(message,1,(255,0,0))
    window.blit(text,(window.get_width()/2-150,window.get_height()/2))
    pygame.display.update()
    #run=False
    #time.sleep(3)
clock=pygame.time.Clock()#計時器
window=pygame.display.set_mode((1400,800))
pygame.display.set_caption("throw_ball!")
background=pygame.Surface(window.get_size())#畫布
background=background.convert()#可有可無
background.fill((255,255,255))#畫布上色

window.blit(background,(0,0))#把畫布貼在繪圖視窗window上
pygame.display.update()


allsprite=pygame.sprite.Group()#角色群組變數
#playersprite=pygame.sprite.Group()
#enemysprite=pygame.sprite.Group()
#bulletsprite=pygame.sprite.Group()
mid=theObject((0,0,255),window.get_width()/2+(R_1+R_2)/R_1*draw_R_1,window.get_height()-400-(R_1+R_2)/R_1*draw_R_1)
ball=theObject((255,0,0),window.get_width()/2+(R_1+R_2+R_1*sin(theta_1)+R_2*sin(theta_2))/R_1*draw_R_1,window.get_height()-400-(R_1+R_2-R_1*cos(theta_1)-R_2*cos(theta_2))/R_1*draw_R_1)
motor=theObject((0,255,0),window.get_width()/2+(R_1+R_2+R_1*sin(theta_1))/R_1*draw_R_1,window.get_height()-400-((R_1+R_2-R_1*cos(theta_1))/R_1*draw_R_1))
#motor=Object((255,0,0),window.get_width()/2,window.get_height()-30)
pygame.draw.line(background, (0, 0, 0), (window.get_width()/2+(R_1+R_2)/R_1*draw_R_1, window.get_height()-400-(R_1+R_2)/R_1*draw_R_1), (window.get_width()/2+(R_1+R_2+R_1*sin(theta_1))/R_1*draw_R_1, window.get_height()-400-((R_1+R_2-R_1*cos(theta_1))/R_1*draw_R_1)))
pygame.draw.line(background, (0, 0, 0), (window.get_width()/2+(R_1+R_2+R_1*sin(theta_1)+R_2*sin(theta_2))/R_1*draw_R_1, window.get_height()-400-(R_1+R_2-R_1*cos(theta_1)-R_2*cos(theta_2))/R_1*draw_R_1), (window.get_width()/2+(R_1+R_2+R_1*sin(theta_1))/R_1*draw_R_1, window.get_height()-400-((R_1+R_2-R_1*cos(theta_1))/R_1*draw_R_1)))
#playersprite.add(ship)
allsprite.add(ball)
allsprite.add(motor)
allsprite.add(mid)
#point=0#計分
prev_line_x1=window.get_width()/2+(R_1+R_2)/R_1*draw_R_1
prev_line_y1=window.get_height()-400-(R_1+R_2)/R_1*draw_R_1

prev_line_x2=window.get_width()/2+(R_1+R_2+R_1*sin(theta_1))/R_1*draw_R_1
prev_line_y2=window.get_height()-400-(R_1+R_2-R_1*cos(theta_1))/R_1*draw_R_1

prev_line_x3=window.get_width()/2+(R_1+R_2+R_1*sin(theta_1)+R_2*sin(theta_2))/R_1*draw_R_1
prev_line_y3=window.get_height()-400-(R_1+R_2-R_1*cos(theta_1)-R_2*cos(theta_2))/R_1*draw_R_1

prev_line_x4=window.get_width()/2+(R_1+R_2+R_1*sin(theta_1))/R_1*draw_R_1
prev_line_y4=window.get_height()-400-((R_1+R_2-R_1*cos(theta_1))/R_1*draw_R_1)



playing=False#playing true代表球正在動
run=True#run false代表程式結束
n=0
while run:
    n+=1
    clock.tick(int(draw_delta_t*1000))
    for event in pygame.event.get():
        if event.type==pygame.QUIT:#使用者者按x結束視窗
            run=False#跳出pygame
    button=pygame.key.get_pressed()
    #按下滑鼠左鍵開始遊戲（求開始動）
    if button[pygame.K_s]:
        playing=True
    if playing:
        window.blit(background,(0,0))#把background在貼到window上 等於清空視窗
        #button2=pygame.mouse.get_pressed()
        pygame.draw.line(background,(255,255,255),(prev_line_x1,prev_line_y1),(prev_line_x2,prev_line_y2))
        pygame.draw.line(background,(255,255,255),(prev_line_x3,prev_line_y3),(prev_line_x4,prev_line_y4))
        if button[pygame.K_0]:
            do_action(0,False)
        elif button[pygame.K_1]:
            do_action(1,False)
        elif button[pygame.K_2]:
            do_action(2,False)
        elif button[pygame.K_3]:
            do_action(3,False)
        elif button[pygame.K_4]:
            do_action(4,False)
        elif button[pygame.K_5]:
            do_action(5,False)
        elif button[pygame.K_6]:
            do_action(6,False)
        elif button[pygame.K_7]:
            do_action(7,False)
        elif button[pygame.K_8]:
            do_action(8,False)
        elif button[pygame.K_9]:
            a,shoot_dis,b=do_action(9,False)
            playing=False
            print("shoot distance: "+str(shoot_dis))
            output_path='./data/data'+str(data_file_num)+'.txt'
            fd = open(output_path,'w',encoding='utf-8')
            for i in range(len(observation_array)):
                for j in range(len(observation_array[i])):
                    fd.write(str(observation_array[i][j])+' ')
            fd.write('\n')
            for i in range(len(action_array)):
                fd.write(str(action_array[i])+' ')
            fd.write('\n')
            for i in range(len(reward_array)):
                fd.write(str(reward_array[i])+' ')
            print("observation_array"+str(len(observation_array)))
            print("action_array"+str(len(action_array)))
            print("reward_array"+str(len(reward_array)))
            observation_array=[]
            action_array=[]
            reward_array=[]
            fd.close()
            data_file_num+=1
            #fd2 = open('./data1.txt','r',encoding='utf-8')
            #tmp=fd2.read()
            #tmp2=tmp.split()
        else:
            do_action(4,False)
        
        motor.update(window.get_width()/2+(R_1+R_2+R_1*sin(theta_1))/R_1*draw_R_1,window.get_height()-400-((R_1+R_2-R_1*cos(theta_1))/R_1*draw_R_1))
        """+((R_1+R_2-R_1*cos(theta_1))/R_1*draw_R_1)"""
        ball.update(window.get_width()/2+(R_1+R_2+R_1*sin(theta_1)+R_2*sin(theta_2))/R_1*draw_R_1,window.get_height()-400-(R_1+R_2-R_1*cos(theta_1)-R_2*cos(theta_2))/R_1*draw_R_1)
        """+(R_1+R_2-R_1*cos(theta_1)-R_2*cos(theta_2))/R_1*draw_R_1"""
        pygame.draw.line(background, (0, 0, 0), (window.get_width()/2+(R_1+R_2)/R_1*draw_R_1, window.get_height()-400-(R_1+R_2)/R_1*draw_R_1), (window.get_width()/2+(R_1+R_2+R_1*sin(theta_1))/R_1*draw_R_1, window.get_height()-400-((R_1+R_2-R_1*cos(theta_1))/R_1*draw_R_1)))
        pygame.draw.line(background, (0, 0, 0), (window.get_width()/2+(R_1+R_2+R_1*sin(theta_1)+R_2*sin(theta_2))/R_1*draw_R_1, window.get_height()-400-(R_1+R_2-R_1*cos(theta_1)-R_2*cos(theta_2))/R_1*draw_R_1), (window.get_width()/2+(R_1+R_2+R_1*sin(theta_1))/R_1*draw_R_1, window.get_height()-400-((R_1+R_2-R_1*cos(theta_1))/R_1*draw_R_1)))
        
        prev_line_x1=window.get_width()/2+(R_1+R_2)/R_1*draw_R_1
        prev_line_y1=window.get_height()-400-(R_1+R_2)/R_1*draw_R_1
        prev_line_x2=window.get_width()/2+(R_1+R_2+R_1*sin(theta_1))/R_1*draw_R_1
        prev_line_y2=window.get_height()-400-(R_1+R_2-R_1*cos(theta_1))/R_1*draw_R_1
        prev_line_x3=window.get_width()/2+(R_1+R_2+R_1*sin(theta_1)+R_2*sin(theta_2))/R_1*draw_R_1
        prev_line_y3=window.get_height()-400-(R_1+R_2-R_1*cos(theta_1)-R_2*cos(theta_2))/R_1*draw_R_1
        prev_line_x4=window.get_width()/2+(R_1+R_2+R_1*sin(theta_1))/R_1*draw_R_1
        prev_line_y4=window.get_height()-400-((R_1+R_2-R_1*cos(theta_1))/R_1*draw_R_1)
        window.blit(background,(0,0))
        
    allsprite.draw(window)
    msgscore="energy: "+str(10)
    msgscoredisplay=font.render(msgscore,5,(255,0,0))
    window.blit(msgscoredisplay, (window.get_width()-350,0))
    pygame.display.update()
window.blit(background,(0,0))#把background在貼到window上 等於清空視窗
gameover('GGWP')
#msgscore="score: "+str(point)
msgscoredisplay=font.render(msgscore,5,(255,0,0))
window.blit(msgscoredisplay, (window.get_width()-300,0))
allsprite.draw(window)
pygame.display.update()
time.sleep(4)
