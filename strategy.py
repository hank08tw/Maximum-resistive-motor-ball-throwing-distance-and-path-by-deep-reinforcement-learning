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
plt.subplot(211)
x=[[1,1],[3,4]]
plt.scatter(x[:,1],x[:,0])
plt.show()
"""
import numpy as np
import pandas as pd
import tensorflow as tf


matplotlib.style.use('ggplot')

#最後svg動畫和theta vs w的圖會畫這個episode的結果
#print_this_episode_number=736#1423
count=0
#能擺動的最大力矩
L_m=1#1
#現在這個state的力矩大小 可以是L_m,0,-L_m
L=0
#現在的角度（是弧度）
theta=0
#現在的角速度
w=0
#現在的角加速度
alpha=0
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
R=0.5#0.5
#畫圖的木棒長度
draw_R=50
#球的質量
m=0.5
#阻力係數k*w
k=0.01
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
energy_diff_coefficient=1#1.1 (10 0.38) (1 0.36)  (0.1 0.33)          
#算法跑幾個episode
num_episode=5000
# reproducible
np.random.seed(1)
tf.set_random_seed(1)


theta_box=[]
w_box=[]
alpha_box=[]
moment=[]


action_space=['forward','dormant','backward','throw']
image = svgwrite.Drawing('strategy.svg', size=(1024,1024))
circle=image.add(image.circle((100+draw_R,0),5))
image.add(image.line((0, 400-draw_R), (1024, 400-draw_R), stroke=svgwrite.rgb(0, 0, 0, '%')))

in_circle_line_num=10
in_circle_line=[]

for i in range(in_circle_line_num):
    in_circle_line.append(image.add(image.circle((100+draw_R,0),1)))


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
    """
    for tmp_t in range(0,math.floor(time),delta_t):
        s_old=y+v*math.sin(theta)*tmp_t-0.5*g*tmp_t*tmp_t
        s_new=y+v*math.sin(theta)*(tmp_t+delta_t)-0.5*g*(tmp_t+delta_t)*(tmp_t+delta_t)
        path = 'M' + str(x+v*math.cos(theta)*tmp_t) + ',' + str(-s_old+400) + ' L ' + str(x+v*math.cos(theta)*(tmp_t+delta_t)) + ',' + str(-s_new+400)
        circle.add(image.animateMotion(path=path,dur="0.02s", begin=str(i/50)+"s",fill="freeze"))
        i=i+1
    """
    return v*math.cos(theta)*time
#%matplotlib inline


def do_action(action,print_this_episode):
    global L_m,L,theta,w,alpha,g,delta_t,R,m,k,k2,i,max_x_num,max_x_val,action_space,count,x_box,y_box,energy_coefficient,last_coefficient,energy_diff_coefficient,total_time,max_total_time
    prev_x=R*sin(theta)
    prev_y=R-R*cos(theta)
    prev_x_in=R*sin(theta)
    prev_y_in=R*cos(theta)
    prev_energy=m*g*prev_y+0.5*m*R*R*w*w
    total_time+=delta_t
    if total_time>=max_total_time:
        print('time out!')
        return np.array([math.sin(theta),math.cos(theta),w]),0,True
    elif action==3:
        #tmp=shoot_the_ball(prev_x,prev_y,R*w*cos(theta),R*w*sin(theta),0,9.98,print_this_episode)
        tmp=shoot_the_ball2(prev_x,prev_y,R*w,theta,print_this_episode)+R*math.sin(theta)
        print('distance: '+str(tmp))
        if tmp>max_x_val:
            max_x_val=tmp
            max_x_num=count
        return np.array([math.sin(theta),math.cos(theta),w]),tmp*last_coefficient,True
    elif action==0:
        L=L_m
        alpha=(L-R*math.sin(theta)*m*g)/(m*R*R)
        w=w+alpha*delta_t-k*w
    elif action==2:
        L=-L_m
        alpha=(L-R*math.sin(theta)*m*g)/(m*R*R)
        w=w+alpha*delta_t-k*w
    elif action==1:
        L=0
        alpha=(L-R*math.sin(theta)*m*g)/(m*R*R)
        w=w+alpha*delta_t-k2*w
    else:
        print("action not exist!!!")
    theta=w*delta_t+theta
    cur_x=R*sin(theta)
    cur_y=R-R*cos(theta)
    cur_x_in=R*sin(theta)
    cur_y_in=R*cos(theta)
    energy=m*g*cur_y+0.5*m*R*R*w*w
    energy_diff=energy-prev_energy
    
    if print_this_episode:
        print('******************')
        #print('i: '+str(i))
        #path = 'M' + str(prev_x) + ',' + str(-prev_y+400) + ' L ' + str(cur_x) + ',' + str(-cur_y+400)
        for _ in range(0,in_circle_line_num):
            tmp_path= 'M'+ str((prev_x_in*_/in_circle_line_num)/R*draw_R) + ',' + str(-((R-prev_y_in*_/in_circle_line_num)/R*draw_R)+400) + ' L ' + str((cur_x_in*_/in_circle_line_num)/R*draw_R) + ',' + str(-(R-cur_y_in*_/in_circle_line_num)/R*draw_R+400)
            in_circle_line[_].add(image.animateMotion(path=tmp_path,dur=str(draw_delta_t)+"s", begin=str(i*draw_delta_t)+"s",fill="freeze"))
        path = 'M' + str(prev_x/R*draw_R) + ',' + str(-prev_y/R*draw_R+400) + ' L ' + str(cur_x/R*draw_R) + ',' + str(-cur_y/R*draw_R+400)
        circle.add(image.animateMotion(path=path,dur=str(draw_delta_t)+"s", begin=str(i*draw_delta_t)+"s",fill="freeze"))
        image.add(image.line((0, -draw_R+400), (1024, -draw_R+400), stroke=svgwrite.rgb(0, 0, 0, '%')))
        #image.add(image.line((R, 0), (R, 1024), stroke=svgwrite.rgb(0, 0, 0, '%')))
        pos_theta=theta*57.29577951
        print('energy and coefficient: '+str(energy*energy_coefficient))
        print("energy_diff and coefficient: "+str(energy_diff*energy_diff_coefficient))
        
        """
        while pos_theta<0:
            pos_theta=pos_theta+360
        while pos_theta>=360:
            pos_theta=pos_theta-360
        """
        theta_box.append(pos_theta)
        w_box.append(w)
        alpha_box.append(alpha)
        moment.append(L)
        print('x: '+str(cur_x))
        print('y: '+str(cur_y))
        #image.save()
        i=i+1
    #cv2.circle(img, (math.floor(10+cur_x), math.floor(255-cur_y)), 5, (255, 255, 255), 1)
    #cv2.imshow('Image', img)
    #cv2.waitKey(0)    
    return np.array([math.sin(theta),math.cos(theta),w]),energy_diff_coefficient*energy_diff,False

#action_space=['forward','dormant','backward','throw']
action_array=[]
for i in range(100):
    action_array.append(0)
for i in range(200):
    action_array.append(2)
for i in range(200):
    action_array.append(0)
action_array.append(3)
print_this_episode=True
for i in range(len(action_array)):
    do_action(action_array[i],print_this_episode)
title('x axis is theta, y axis is w')
plot(theta_box, w_box, 'r*')
plot(theta_box[:], w_box[:])
show()
title('x axis is theta, y axis is alpha')
plot(theta_box, alpha_box, 'r*')
plot(theta_box[:], alpha_box[:])
show()
title('x axis is time, y axis is moment')
plot(moment,'r*')
show()
image.save()
