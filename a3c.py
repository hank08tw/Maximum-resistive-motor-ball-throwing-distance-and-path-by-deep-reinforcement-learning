"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.
The cart pole example. Policy is oscillated.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
tensorflow 1.0
gym 0.8.0
"""
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

import tensorflow as tf


np.random.seed(1)
tf.set_random_seed(1)  # reproducible

# Superparameters
#OUTPUT_GRAPH = True
#MAX_EPISODE = 3000
#DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
#MAX_EP_STEPS = 1000   # maximum time step in one episode
#RENDER = True  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

#env = gym.make('CartPole-v0')
#env.seed(1)  # reproducible
#env = env.unwrapped

#最後svg動畫和theta vs w的圖會畫這個episode的結果
print_this_episode_number=5000
count=0
#能擺動的最大力矩
L_m=5#1
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
R=0.3#0.5
#畫圖的木棒長度
draw_R=50
#球的質量
m=0.3
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
energy_diff_coefficient=10#1.1 (10 0.38) (1 0.36)  (0.1 0.33)          
#算法跑幾個episode
num_episode=5000


#用來畫圖 儲存每個時間點theta跟w的值
theta_box = []
w_box = []
alpha_box= []
moment=[]

action_space=['forward','dormant','backward','throw']
image = svgwrite.Drawing('a3c_animation.svg', size=(1024,1024))
circle=image.add(image.circle((100+draw_R,0),5))
image.add(image.line((0, 400-draw_R), (1024, 400-draw_R), stroke=svgwrite.rgb(0, 0, 0, '%')))

in_circle_line_num=10
in_circle_line=[]

for i in range(in_circle_line_num):
    in_circle_line.append(image.add(image.circle((100+draw_R*2,0),1)))


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
    prev_x=R+R*sin(theta)
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
    cur_x=R+R*sin(theta)
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



N_F = 3
N_A = 4


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error


sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())

#if OUTPUT_GRAPH:
#    tf.summary.FileWriter("logs/", sess.graph)

for episode in range(num_episode):
    s = np.array([0,1,0])
    t = 0
    track_r = []
    if print_this_episode_number==episode:
        print_this_episode=True
    else:
        print_this_episode=False
    while True:
        #if RENDER: env.render()
        a = actor.choose_action(s)

        s_, r, done = do_action(a,print_this_episode)

        if done: r = -20

        track_r.append(r)

        td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

        s = s_
        t += 1

        if done:#or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            #if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", episode, "  reward:", int(running_reward))
            break
    #print('distance: '+str())
    print('max_num: '+str(max_x_num))
    print('max_val: '+str(max_x_val))
    count+=1
    if episode==print_this_episode_number:
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
    theta=0
    w=0
    alpha=0
    total_time=0
    theta_box=[]
    w_box=[]
    alpha_box=[]
    print('max_x_num: '+ str(max_x_num))
    print('max_x_val: '+str(max_x_val))