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
"""
f1 = plt.figure(1)
plt.subplot(211) artless
x=[[1,1],[3,4]]
plt.scatter(x[:,1],x[:,0])
plt.show()
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

matplotlib.style.use('ggplot')

#最後svg動畫和theta vs w的圖會畫這個episode的結果
print_this_episode_number=4999
count=0
#上面能擺動的最大力矩
L_m1=2.2
#中間電機能擺動的最大力矩
L_m2=1.5
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
max_total_time=50000000.0
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
ii=0
#目前找到最大路徑是哪個episode
max_x_num=-1
#最大路徑episode丟出的距離
max_x_val=-2147483647
#即時的reward，此係數乘上總能量當作reward
energy_coefficient=1
#最後的reward，此係數乘上拋出的距離當作最後的reward
last_coefficient=1
#即時的reward，此係數呈上能量差當作reward
energy_diff_coefficient=0.01#0.1 1.16 1 421
#算法跑幾個episode
num_episode=10000
# reproducible
np.random.seed(1)
tf.set_random_seed(1)

#
tmp_reward=0
#
limit=10
no_change=0
last_action=-1

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


image = svgwrite.Drawing('my_policy_gradient_animation_2.svg', size=(1024,1024))

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
    """
    if print_this_episode:
        tmp_t=0
        while tmp_t<=math.floor(time) and x+v*math.cos(theta)*tmp_t<1000:
            s_old=y+v*math.sin(theta)*tmp_t-0.5*g*tmp_t*tmp_t
            s_new=y+v*math.sin(theta)*(tmp_t+delta_t)-0.5*g*(tmp_t+delta_t)*(tmp_t+delta_t)
            #path = 'M' + str(x+v*math.cos(theta)*tmp_t) + ',' + str(-s_old+400) + ' L ' + str(x+v*math.cos(theta)*(tmp_t+delta_t)) + ',' + str(-s_new+400)
            #circle.add(image.animateMotion(path=path,dur=str(draw_delta_t)+"s", begin=str(i*draw_delta_t)+"s",fill="freeze"))
            #i=i+1
            tmp_t=tmp_t+delta_t
    """
    return v*math.cos(theta)*time
#%matplotlib inline


def do_action(action,print_this_episode):
    #print('math.pi: '+str(math.pi))
    global L_m1,L_m2,L_1,L_2,theta_1,theta_2,w_1,w_2,alpha_1,alpha_2,g,delta_t,R_1,R_2,m_1,m_2,k,k2,i,max_x_num,max_x_val,action_space,count,theta_box_1,theta_box_2,w_box_1,w_box_2,alpha_box_1,alpha_box_2,energy_coefficient,last_coefficient,energy_diff_coefficient,total_time,max_total_time,in_circle_line,in_circle_line_num,out_circle_line,out_circle_line_num,tmp_reward
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
        return np.array([sin(theta_1),cos(theta_1),w_1,sin(theta_2),cos(theta_2),w_2]),0,True#0.01*total_time
    elif action==9:
        print('9,shoot the ball')
        tmp=shoot_the_ball2(prev_x,prev_y,v_3,theta_2,print_this_episode)+R_1*sin(theta_1)+R_2*math.sin(theta_2)
        print('distance: '+str(tmp))
        if tmp>max_x_val:
            max_x_val=tmp
            max_x_num=count
        #沒有負的reward
        if tmp<=0:
            tmp=-10
        return np.array([sin(theta_1),cos(theta_1),w_1,sin(theta_2),cos(theta_2),w_2]),tmp*last_coefficient,True
    elif action==0:
        print('0--------------')
        L_1=-L_m1
        L_2=-L_m2
        #print(m_1*L_1*L_1+m_2*abs(L_1*L_1+L_2*L_2-2*L_1*L_2*cos(math.pi-(theta_1-theta_2))))
        alpha_1=(L_1-m_1*g*sin(theta_1)*R_1-m_2*g*(R_1*sin(theta_1)+R_2*sin(theta_2)))/(m_1*R_1*R_1+m_2*(R_1*R_1+R_2*R_2-2*R_1*R_2*cos(math.pi-(theta_1-theta_2))))
        alpha_2=(L_2-m_2*g*R_2*sin(theta_2))/(m_2*R_2*R_2)+alpha_1
        w_1=w_1+alpha_1*delta_t-k*w_1
        w_2=w_2+alpha_2*delta_t-k*w_2
    elif action==1:
        print('1--------------')
        L_1=-L_m1
        L_2=0
        #print(m_1*L_1*L_1+m_2*abs(L_1*L_1+L_2*L_2-2*L_1*L_2*cos(math.pi-(theta_1-theta_2))))
        alpha_1=(L_1-m_1*g*sin(theta_1)*R_1-m_2*g*(R_1*sin(theta_1)+R_2*sin(theta_2)))/(m_1*R_1*R_1+m_2*(R_1*R_1+R_2*R_2-2*R_1*R_2*cos(math.pi-(theta_1-theta_2))))
        alpha_2=(L_2-m_2*g*R_2*sin(theta_2))/(m_2*R_2*R_2)+alpha_1
        w_1=w_1+alpha_1*delta_t-k*w_1
        w_2=w_2+alpha_2*delta_t-k2*w_2
    elif action==2:
        print('2--------------')
        L_1=-L_m1
        L_2=L_m2
        #print(m_1*L_1*L_1+m_2*abs(L_1*L_1+L_2*L_2-2*L_1*L_2*cos(math.pi-(theta_1-theta_2))))
        alpha_1=(L_1-m_1*g*sin(theta_1)*R_1-m_2*g*(R_1*sin(theta_1)+R_2*sin(theta_2)))/(m_1*R_1*R_1+m_2*(R_1*R_1+R_2*R_2-2*R_1*R_2*cos(math.pi-(theta_1-theta_2))))
        alpha_2=(L_2-m_2*g*R_2*sin(theta_2))/(m_2*R_2*R_2)+alpha_1
        w_1=w_1+alpha_1*delta_t-k*w_1
        w_2=w_2+alpha_2*delta_t-k*w_2
    elif action==3:
        print('3--------------')
        L_1=0
        L_2=-L_m2
        #print(m_1*L_1*L_1+m_2*abs(L_1*L_1+L_2*L_2-2*L_1*L_2*cos(math.pi-(theta_1-theta_2))))
        alpha_1=(L_1-m_1*g*sin(theta_1)*R_1-m_2*g*(R_1*sin(theta_1)+R_2*sin(theta_2)))/(m_1*L_1*L_1+m_2*(R_1*R_1+R_2*R_2-2*R_1*R_2*cos(math.pi-(theta_1-theta_2))))
        alpha_2=(L_2-m_2*g*R_2*sin(theta_2))/(m_2*R_2*R_2)+alpha_1
        w_1=w_1+alpha_1*delta_t-k2*w_1
        w_2=w_2+alpha_2*delta_t-k*w_2
    elif action==4:
        print('4--------------')
        L_1=0
        L_2=0
        """
        print('0000000000m_1'+str(m_1))
        print('===m_2'+str(m_2))
        print('===L_1'+str(L_1))
        print('===L_2'+str(L_2))
        print('===theta_1'+str(theta_1))
        print('===R_1'+str(R_1))
        print((m_1*L_1*L_1+m_2*abs(L_1*L_1+L_2*L_2-2*L_1*L_2*cos(math.pi-(theta_1-theta_2)))))
        """
        alpha_1=(L_1-m_1*g*sin(theta_1)*R_1-m_2*g*(R_1*sin(theta_1)+R_2*sin(theta_2)))/(m_1*L_1*L_1+m_2*(R_1*R_1+R_2*R_2-2*R_1*R_2*cos(math.pi-(theta_1-theta_2))))
        alpha_2=(L_2-m_2*g*R_2*sin(theta_2))/(m_2*R_2*R_2)+alpha_1
        w_1=w_1+alpha_1*delta_t-k2*w_1
        w_2=w_2+alpha_2*delta_t-k2*w_2
    elif action==5:
        print('5--------------')
        L_1=0
        L_2=L_m2
        #print(m_1*L_1*L_1+m_2*abs(L_1*L_1+L_2*L_2-2*L_1*L_2*cos(math.pi-(theta_1-theta_2))))
        alpha_1=(L_1-m_1*g*sin(theta_1)*R_1-m_2*g*(R_1*sin(theta_1)+R_2*sin(theta_2)))/(m_1*L_1*L_1+m_2*(R_1*R_1+R_2*R_2-2*R_1*R_2*cos(math.pi-(theta_1-theta_2))))
        alpha_2=(L_2-m_2*g*R_2*sin(theta_2))/(m_2*R_2*R_2)+alpha_1
        w_1=w_1+alpha_1*delta_t-k2*w_1
        w_2=w_2+alpha_2*delta_t-k*w_2
    elif action==6:
        print('6--------------')
        L_1=L_m1
        L_2=-L_m2
        #print(m_1*L_1*L_1+m_2*abs(L_1*L_1+L_2*L_2-2*L_1*L_2*cos(math.pi-(theta_1-theta_2))))
        alpha_1=(L_1-m_1*g*sin(theta_1)*R_1-m_2*g*(R_1*sin(theta_1)+R_2*sin(theta_2)))/(m_1*L_1*L_1+m_2*(R_1*R_1+R_2*R_2-2*R_1*R_2*cos(math.pi-(theta_1-theta_2))))
        alpha_2=(L_2-m_2*g*R_2*sin(theta_2))/(m_2*R_2*R_2)+alpha_1
        w_1=w_1+alpha_1*delta_t-k*w_1
        w_2=w_2+alpha_2*delta_t-k*w_2
    elif action==7:
        print('7--------------')
        L_1=L_m1
        L_2=0
        #print(m_1*L_1*L_1+m_2*abs(L_1*L_1+L_2*L_2-2*L_1*L_2*cos(math.pi-(theta_1-theta_2))))
        alpha_1=(L_1-m_1*g*sin(theta_1)*R_1-m_2*g*(R_1*sin(theta_1)+R_2*sin(theta_2)))/(m_1*L_1*L_1+m_2*(R_1*R_1+R_2*R_2-2*R_1*R_2*cos(math.pi-(theta_1-theta_2))))
        alpha_2=(L_2-m_2*g*R_2*sin(theta_2))/(m_2*R_2*R_2)+alpha_1
        w_1=w_1+alpha_1*delta_t-k*w_1
        w_2=w_2+alpha_2*delta_t-k2*w_2
    elif action==8:
        print('8--------------')
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
    
    if print_this_episode:
        print('******************')
        #line_1=image.add(image.line(((R_1+R_2)/R_1*draw_R_1, -(R_1+R_2)/R_1*draw_R_1+400), ((R_1+R_2+prev_x_in)/R_1*draw_R_1, -((R_1+R_2-prev_y_in)/R_1*draw_R_1)+400), stroke=svgwrite.rgb(0, 0, 0, '%')))
        
        for _ in range(0,in_circle_line_num):
            tmp_path= 'M'+ str((R_1+R_2+prev_x_in*_/in_circle_line_num)/R_1*draw_R_1) + ',' + str(-((R_1+R_2-prev_y_in*_/in_circle_line_num)/R_1*draw_R_1)+400) + ' L ' + str((R_1+R_2+cur_x_in*_/in_circle_line_num)/R_1*draw_R_1) + ',' + str(-(R_1+R_2-cur_y_in*_/in_circle_line_num)/R_1*draw_R_1+400)
            in_circle_line[_].add(image.animateMotion(path=tmp_path,dur=str(draw_delta_t)+"s", begin=str(i*draw_delta_t)+"s",fill="freeze"))
        for _ in range(0,out_circle_line_num):
            tmp_path2= 'M'+ str((R_1+R_2+prev_x_in+prev_x_out*_/out_circle_line_num)/R_1*draw_R_1) + ',' + str(-(R_1+R_2-prev_y_in-prev_y_out*_/out_circle_line_num)/R_1*draw_R_1+400) + ' L ' + str((R_1+R_2+cur_x_in+cur_x_out*_/out_circle_line_num)/R_1*draw_R_1) + ',' + str(-(R_1+R_2-cur_y_in-cur_y_out*_/out_circle_line_num)/R_1*draw_R_1+400)
            out_circle_line[_].add(image.animateMotion(path=tmp_path2,dur=str(draw_delta_t)+"s", begin=str(i*draw_delta_t)+"s",fill="freeze"))
        
        """
        in_circle_line_1_path='M'+ str((R_1+R_2+prev_x_in*1/in_circle_line_num)/R_1*draw_R_1) + ',' + str(-((R_1+R_2-prev_y_in*1/in_circle_line_num)/R_1*draw_R_1)+400) + ' L ' + str((R_1+R_2+cur_x_in*1/in_circle_line_num)/R_1*draw_R_1) + ',' + str(-(R_1+R_2-cur_y_in*1/in_circle_line_num)/R_1*draw_R_1+400)
        in_circle_line_1.add(image.animateMotion(path=in_circle_line_1_path,dur=str(draw_delta_t)+"s",begin=str(i*draw_delta_t)+"s",fill="freeze"))
        in_circle_line_2_path='M'+ str((R_1+R_2+prev_x_in*2/in_circle_line_num)/R_1*draw_R_1) + ',' + str(-((R_1+R_2-prev_y_in*2/in_circle_line_num)/R_1*draw_R_1)+400) + ' L ' + str((R_1+R_2+cur_x_in*1/in_circle_line_num)/R_1*draw_R_1) + ',' + str(-(R_1+R_2-cur_y_in*1/in_circle_line_num)/R_1*draw_R_1+400)
        in_circle_line_2.add(image.animateMotion(path=in_circle_line_1_path,dur=str(draw_delta_t)+"s",begin=str(i*draw_delta_t)+"s",fill="freeze"))
        in_circle_line_3_path='M'+ str((R_1+R_2+prev_x_in*3/in_circle_line_num)/R_1*draw_R_1) + ',' + str(-((R_1+R_2-prev_y_in*3/in_circle_line_num)/R_1*draw_R_1)+400) + ' L ' + str((R_1+R_2+cur_x_in*1/in_circle_line_num)/R_1*draw_R_1) + ',' + str(-(R_1+R_2-cur_y_in*1/in_circle_line_num)/R_1*draw_R_1+400)
        in_circle_line_3.add(image.animateMotion(path=in_circle_line_1_path,dur=str(draw_delta_t)+"s",begin=str(i*draw_delta_t)+"s",fill="freeze"))
        in_circle_line_4_path='M'+ str((R_1+R_2+prev_x_in*4/in_circle_line_num)/R_1*draw_R_1) + ',' + str(-((R_1+R_2-prev_y_in*4/in_circle_line_num)/R_1*draw_R_1)+400) + ' L ' + str((R_1+R_2+cur_x_in*1/in_circle_line_num)/R_1*draw_R_1) + ',' + str(-(R_1+R_2-cur_y_in*1/in_circle_line_num)/R_1*draw_R_1+400)
        in_circle_line_4.add(image.animateMotion(path=in_circle_line_1_path,dur=str(draw_delta_t)+"s",begin=str(i*draw_delta_t)+"s",fill="freeze"))
        in_circle_line_5_path='M'+ str((R_1+R_2+prev_x_in*5/in_circle_line_num)/R_1*draw_R_1) + ',' + str(-((R_1+R_2-prev_y_in*5/in_circle_line_num)/R_1*draw_R_1)+400) + ' L ' + str((R_1+R_2+cur_x_in*1/in_circle_line_num)/R_1*draw_R_1) + ',' + str(-(R_1+R_2-cur_y_in*1/in_circle_line_num)/R_1*draw_R_1+400)
        in_circle_line_5.add(image.animateMotion(path=in_circle_line_1_path,dur=str(draw_delta_t)+"s",begin=str(i*draw_delta_t)+"s",fill="freeze"))
        in_circle_line_6_path='M'+ str((R_1+R_2+prev_x_in*6/in_circle_line_num)/R_1*draw_R_1) + ',' + str(-((R_1+R_2-prev_y_in*6/in_circle_line_num)/R_1*draw_R_1)+400) + ' L ' + str((R_1+R_2+cur_x_in*1/in_circle_line_num)/R_1*draw_R_1) + ',' + str(-(R_1+R_2-cur_y_in*1/in_circle_line_num)/R_1*draw_R_1+400)
        in_circle_line_6.add(image.animateMotion(path=in_circle_line_1_path,dur=str(draw_delta_t)+"s",begin=str(i*draw_delta_t)+"s",fill="freeze"))
        in_circle_line_7_path='M'+ str((R_1+R_2+prev_x_in*7/in_circle_line_num)/R_1*draw_R_1) + ',' + str(-((R_1+R_2-prev_y_in*7/in_circle_line_num)/R_1*draw_R_1)+400) + ' L ' + str((R_1+R_2+cur_x_in*1/in_circle_line_num)/R_1*draw_R_1) + ',' + str(-(R_1+R_2-cur_y_in*1/in_circle_line_num)/R_1*draw_R_1+400)
        in_circle_line_7.add(image.animateMotion(path=in_circle_line_1_path,dur=str(draw_delta_t)+"s",begin=str(i*draw_delta_t)+"s",fill="freeze"))
        in_circle_line_8_path='M'+ str((R_1+R_2+prev_x_in*8/in_circle_line_num)/R_1*draw_R_1) + ',' + str(-((R_1+R_2-prev_y_in*8/in_circle_line_num)/R_1*draw_R_1)+400) + ' L ' + str((R_1+R_2+cur_x_in*1/in_circle_line_num)/R_1*draw_R_1) + ',' + str(-(R_1+R_2-cur_y_in*1/in_circle_line_num)/R_1*draw_R_1+400)
        in_circle_line_8.add(image.animateMotion(path=in_circle_line_1_path,dur=str(draw_delta_t)+"s",begin=str(i*draw_delta_t)+"s",fill="freeze"))
        in_circle_line_9_path='M'+ str((R_1+R_2+prev_x_in*9/in_circle_line_num)/R_1*draw_R_1) + ',' + str(-((R_1+R_2-prev_y_in*9/in_circle_line_num)/R_1*draw_R_1)+400) + ' L ' + str((R_1+R_2+cur_x_in*1/in_circle_line_num)/R_1*draw_R_1) + ',' + str(-(R_1+R_2-cur_y_in*1/in_circle_line_num)/R_1*draw_R_1+400)
        in_circle_line_9.add(image.animateMotion(path=in_circle_line_1_path,dur=str(draw_delta_t)+"s",begin=str(i*draw_delta_t)+"s",fill="freeze"))
        in_circle_line_10_path='M'+ str((R_1+R_2+prev_x_in*10/in_circle_line_num)/R_1*draw_R_1) + ',' + str(-((R_1+R_2-prev_y_in*10/in_circle_line_num)/R_1*draw_R_1)+400) + ' L ' + str((R_1+R_2+cur_x_in*1/in_circle_line_num)/R_1*draw_R_1) + ',' + str(-(R_1+R_2-cur_y_in*1/in_circle_line_num)/R_1*draw_R_1+400)
        in_circle_line_10.add(image.animateMotion(path=in_circle_line_1_path,dur=str(draw_delta_t)+"s",begin=str(i*draw_delta_t)+"s",fill="freeze"))
        """
        mid_path = 'M' + str((R_1+R_2)/R_1*draw_R_1) + ',' + str(-(R_1+R_2)/R_1*draw_R_1+400) + ' L ' + str((R_1+R_2)/R_1*draw_R_1) + ',' + str(-(R_1+R_2)/R_1*draw_R_1+400)
        mid.add(image.animateMotion(path=mid_path,dur=str(draw_delta_t)+"s", begin=str(i*draw_delta_t)+"s",fill="freeze"))
        path = 'M' + str((R_1+R_2+prev_x_in)/R_1*draw_R_1) + ',' + str(-((R_1+R_2-prev_y_in)/R_1*draw_R_1)+400) + ' L ' + str((R_1+R_2+cur_x_in)/R_1*draw_R_1) + ',' + str(-(R_1+R_2-cur_y_in)/R_1*draw_R_1+400)
        circle.add(image.animateMotion(path=path,dur=str(draw_delta_t)+"s", begin=str(i*draw_delta_t)+"s",fill="freeze"))
        path2 = 'M' + str((R_1+R_2+prev_x_in+prev_x_out)/R_1*draw_R_1) + ',' + str(-(R_1+R_2-prev_y_in-prev_y_out)/R_1*draw_R_1+400) + ' L ' + str((R_1+R_2+cur_x_in+cur_x_out)/R_1*draw_R_1) + ',' + str(-(R_1+R_2-cur_y_in-cur_y_out)/R_1*draw_R_1+400)
        circle2.add(image.animateMotion(path=path2,dur=str(draw_delta_t)+"s", begin=str(i*draw_delta_t)+"s",fill="freeze"))
        image.add(image.line((0, -(draw_R_1+draw_R_2)+400), (1024, -(draw_R_1+draw_R_2)+400), stroke=svgwrite.rgb(0, 0, 0, '%')))
        
        """
        path = 'M' + str(R_1+R_2+prev_x_in/R_1*draw_R_1) + ',' + str(-(R_1+R_2-prev_y_in/R_1*draw_R_1)+400) + ' L ' + str(R_1+R_2+cur_x_in/R_1*draw_R_1) + ',' + str(-(R_1+R_2-cur_y_in/R_1*draw_R_1)+400)
        circle.add(image.animateMotion(path=path,dur=str(draw_delta_t)+"s", begin=str(i*draw_delta_t)+"s",fill="freeze"))
        path2 = 'M' + str((R_1+R_2)+prev_x_in/R_1*draw_R_1+prev_x_out/R_2*draw_R_2) + ',' + str(-(R_1+R_2-prev_y_in/R_1*draw_R_1-prev_y_out/R_2*draw_R_2)+400) + ' L ' + str(R_1+R_2+cur_x_in/R_1*draw_R_1+cur_x_out/R_2*draw_R_2) + ',' + str(-(R_1+R_2-cur_y_in/R_1*draw_R_1-cur_y_out/R_2*draw_R_2)+400)
        circle2.add(image.animateMotion(path=path2,dur=str(draw_delta_t)+"s", begin=str(i*draw_delta_t)+"s",fill="freeze"))
        image.add(image.line((0, -(draw_R_1+draw_R_2)+400), (1024, -(draw_R_1+draw_R_2)+400), stroke=svgwrite.rgb(0, 0, 0, '%')))
        """
        pos_theta_1=theta_1*57.29577951
        pos_theta_2=theta_2*57.29577951 
        print('energy and coefficient: '+str(energy*energy_coefficient))
        print("energy_diff and coefficient: "+str(energy_diff*energy_diff_coefficient))
        theta_box_1.append(pos_theta_1)
        theta_box_2.append(pos_theta_2)
        w_box_1.append(w_1)
        w_box_2.append(w_2)
        alpha_box_1.append(alpha_1)
        alpha_box_2.append(alpha_box_2)
        moment_1.append(L_1)
        moment_2.append(L_2)
        x_box_1.append(R_1+R_2+R_1*sin(theta_1))
        y_box_1.append(R_1+R_2-R_1*cos(theta_1))
        x_box_2.append(R_1+R_2+R_1*sin(theta_1)+R_2*sin(theta_2))
        y_box_2.append(R_1+R_2-R_1*cos(theta_1)-R_2*cos(theta_2))
        print('x: '+str(cur_x))
        print('y: '+str(cur_y))
        i=i+1
    #cv2.circle(img, (math.floor(10+cur_x), math.floor(255-cur_y)), 5, (255, 255, 255), 1)
    #cv2.imshow('Image', img)
    #cv2.waitKey(0)    
    return np.array([math.sin(theta_1),math.cos(theta_1),w_1,sin(theta_2),cos(theta_2),w_2]),energy_diff_coefficient*energy_diff,False






import numpy as np
import tensorflow as tf




class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=True,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # fc1
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,#輸出維度
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,#輸出維度 有幾個神經元
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]}) # (n,) -> (1,n)
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        #print('np.std: '+str(np.std(discounted_ep_rs)))
        discounted_ep_rs = np.array(discounted_ep_rs, dtype=np.float64)
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        if np.std(discounted_ep_rs)!=0:
            discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs





import gym
#from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

RL = PolicyGradient(
    n_actions=len(action_space),
    n_features=6,
    learning_rate=0.02,
    reward_decay=0.995,
    # output_graph=True,
)
print_this_episode=False
observation = np.array([0,1,0,0,1,0])
step = 0

for episode in range(num_episode):
    #global L_m,L,theta,w,g,delta_t,R,m,k,i,max_x_num,max_x_val,action_space,count,num_episode
    #print('episode: '+str(episode))
    total_time=0
    tmp_reward=0
    if episode==print_this_episode_number:
        print_this_episode=True
    else:
        print_this_episode=False
    observation=np.array([sin(theta_1),cos(theta_1),w_1,sin(theta_2),cos(theta_2),w_2])
    while True:
        #if RENDER: env.render()
        
        if no_change>0:
            action=last_action
            no_change-=1
        else:
            action = RL.choose_action(observation)
            if action!=last_action:
                no_change=limit
                last_action=action
        
        #action = RL.choose_action(observation)

        observation_, reward, done = do_action(action,print_this_episode)     # reward = -1 in all cases
        """
        print('------------------theta_1'+str(theta_1))
        print('theta_2'+str(theta_2))
        print('w_1'+str(w_1))
        print('w_2'+str(w_2))
        print('alpha_1'+str(alpha_1))
        print('alpha_2'+str(alpha_2))
        print('math.pi'+str(math.pi))
        """
        RL.store_transition(observation, action, reward)

        if done:
            # calculate running reward
            ep_rs_sum = sum(RL.ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            #if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering

            print("episode:", episode, "  reward:", int(running_reward))

            vt = RL.learn()  # train
            break
            
        observation = observation_
    count=count+1
    
    if episode==print_this_episode_number:
        """
        title('x axis is theta, y axis is w')
        plot(theta_box_1, w_box_1, 'r*')
        plot(theta_box_1[:], w_box_1[:])
        plot(theta_box_2, w_box_2, 'b*')
        plot(theta_box_2[:], w_box_2[:])
        show()
        """
        """
        title('x axis is theta, y axis is alpha')
        plot(theta_box_1, alpha_box_1, 'r*')
        plot(theta_box_1[:], alpha_box_1[:])
        plot(theta_box_2, alpha_box_2, 'b*')
        plot(theta_box_2[:], alpha_box_2[:])
        show()
        """
        
        title('x axis is time, y axis is moment')
        plot(moment_1,'r*')
        plot(moment_2,'b*')
        show()
        
        """
        title('x axis is x position(m), y axis is y position(m)')
        plot(x_box_1,y_box_1,'r*')
        #plot(x_box_1[:],y_box_1[:])
        plot(x_box_2,y_box_2,'b*')
        #plot(x_box_2[:],y_box_2[:])
        show()
        """
        
        image.save()
        
    theta_1=0
    theta_2=0
    w_1=0
    w_2=0
    alpha_1=0
    alpha_2=0
    theta_box_1=[]
    theta_box_2=[]
    w_box_1=[]
    w_box_2=[]
    alpha_box_1=[]
    alpha_box_2=[]
    x_box_1=[]
    y_box_1=[]
    x_box_2=[]
    y_box_2=[]
    print('max_x_num: '+ str(max_x_num))
    print('max_x_val: '+str(max_x_val))

    
print('change print_this_episode_number to this number to print the max_x_num: '+ str(max_x_num))
print(max_x_val)