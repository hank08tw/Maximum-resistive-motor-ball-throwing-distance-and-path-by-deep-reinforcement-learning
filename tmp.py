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


import numpy as np
img = np.zeros([512, 512, 3])
image = svgwrite.Drawing('animation.svg', size=(1024,1024))
circle=image.add(image.circle((100,0),5))
image.add(image.add(image.line((0, 200), (1024, 200), stroke=svgwrite.rgb(0, 0, 0, '%'))))
image.add(image.line((200, 0), (200, 1024), stroke=svgwrite.rgb(0, 0, 0, '%')))
#cv2.imshow('Image', img)
#cv2.waitKey(0)
L_m=7
L=0
theta=0
w=0
g=9.98
delta_t=0.01
R=100
m=0.01
k=0.0005 
i=0
max_x_num=-1
max_x_val=-2147483647
def shoot_the_ball(x,y,v_x,v_y,a_x,a_y):
    global delta_t,i
    prev_x=-1
    prev_y=-1
    cur_x=x
    cur_y=y
    while True:
        prev_x=cur_x
        prev_y=cur_y
        cur_x=cur_x+delta_t*v_x
        cur_y=cur_y+delta_t*v_y
        #path = 'M' + str(prev_x+200) + ',' + str(-prev_y+400) + ' L ' + str(cur_x) + ',' + str(-cur_y+400)
        #circle.add(image.animateMotion(path=path,dur="0.02s", begin=str(i/50)+"s",fill="freeze"))
        i=i+1
        v_x=v_x+delta_t*a_x
        v_y=v_y-delta_t*a_y
        if cur_y<=0:
            return cur_x
        #print(y)

def step(action):
    global L,theta,w,g,delta_t,R,m,k,i,max_x_num,max_x_val
    prev_x=R+R*sin(theta)
    prev_y=R-R*cos(theta)
    if action=='throw':
        tmp=shoot_the_ball(R*sin(theta),R-R*cos(theta),R*w*cos(theta),R*w*sin(theta),0,9.98)
        print('distance: '+str(tmp+R+R*sin(theta)))
        if tmp+R+R*sin(theta)>max_x_val:
            max_x_val=tmp+R+R*sin(theta)
        return "terminal",tmp-R,True
    elif action=="forward":
        L=L_m
    elif action=="backward":
        L=-L_m
    elif action=="dormant":
        L=0
    else:
        print("action not exist!!!")
    w=w+(L-R*math.sin(theta)*m*g)/(m*R*R)-k*w
    theta=w*delta_t+theta
    cur_x=R+R*sin(theta)
    cur_y=R-R*cos(theta)
    path = 'M' + str(prev_x) + ',' + str(-prev_y+400) + ' L ' + str(cur_x) + ',' + str(-cur_y+400)
    circle.add(image.animateMotion(path=path,dur="0.02s", begin=str(i/50)+"s",fill="freeze"))
    #cv2.circle(img, (math.floor(10+cur_x), math.floor(255-cur_y)), 5, (255, 255, 255), 1)
    #cv2.imshow('Image', img)
    #cv2.waitKey(0)
    i=i+1
    image.save()
    return str(round(theta,1))+"+"+str(round(w,1)),0,False
    
#draw(7,0,0,9.98,0.1,100,0.01,0.05,False,300)
if __name__=="__main__":
    observation="0+0"
    action_space = ['forward', 'dormant', 'backward', 'throw']#forward逆時針，dormant不動，backward順時針，throw拋球
    n_actions = len(action_space)
    RL=QLearningTable(actions=list(action_space))
    for episode in range(1000):
        print('episode: '+str(episode))
        while True:
            action=RL.choose_action(str(observation))
            observation_,reward,done=step(action)
            RL.learn(str(observation),action,reward,str(observation_))
            observation=observation_
            if done:
                break
        theta=0
        w=0
        #cv2.imshow('Image', img)
        #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    q_table_file=RL.q_table
    q_table_file.to_json("./q_table_file.json",)
    print(q_table_file)
"""
"""
from PIL import Image
from pylab import *

#添加标题信息
title('x is theta, y is dot_theta')
#随意给的一些点
x = [100, 100, 400, 400]
y = [200, 500, 200, 500]
 
#使用红色-星状标记需要绘制的点
plot(x, y, 'r*')
plot(x[:], y[:])
show()
"""
a='noob'
globals()
print(globals())
"""