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
print_this_episode_number=4998#1423
count=0
#能擺動的最大力矩
L_m=0.4#1
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
max_total_time=50000000.0
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
energy_diff_coefficient=1#1.1 (10 0.38) (1 0.36)  (0.1 0.33)          
#算法跑幾個episode
num_episode=5000
#
limit=20
no_change=0
last_action=-1


# reproducible
np.random.seed(1)
tf.set_random_seed(1)

#用來畫圖 儲存每個時間點theta跟w的值
theta_box = []
w_box = []
alpha_box= []
moment=[]

action_space=['forward','dormant','backward','throw']
image = svgwrite.Drawing('my_policy_gradient_animation.svg', size=(1024,1024))
circle=image.add(image.circle((100+draw_R,0),5))
image.add(image.line((0, 400-draw_R), (1024, 400-draw_R), stroke=svgwrite.rgb(0, 0, 0, '%')))

in_circle_line_num=10
in_circle_line=[]

for i in range(in_circle_line_num):
    in_circle_line.append(image.add(image.circle((100+draw_R,0),1)))


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
        #new layer
        layer2 = tf.layers.dense(
            inputs=layer,
            units=10,#輸出維度
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='mylayer1'
        )
        # fc2
        all_act = tf.layers.dense(
            inputs=layer2,
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
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        if np.std(discounted_ep_rs)!=0:
            discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs





import gym
#from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

#DISPLAY_REWARD_THRESHOLD = -2000  # renders environment if total episode reward is greater then this threshold
# episode: 154   reward: -10667
# episode: 387   reward: -2009
# episode: 489   reward: -1006
# episode: 628   reward: -502

#RENDER = False  # rendering wastes time

#env = gym.make('MountainCar-v0')
#env.seed(1)     # reproducible, general Policy gradient has high variance
#env = env.unwrapped

#print(env.action_space)
#print(env.observation_space)
#print(env.observation_space.high)
#print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=len(action_space),
    n_features=3,
    learning_rate=0.02,
    reward_decay=0.995,
    # output_graph=True,
)
print_this_episode=False
observation = np.array([0,1,0])
step = 0

for episode in range(num_episode):
    #global L_m,L,theta,w,g,delta_t,R,m,k,i,max_x_num,max_x_val,action_space,count,num_episode
    #print('episode: '+str(episode))
    if episode==print_this_episode_number:
        print_this_episode=True
    else:
        print_this_episode=False
    observation=np.array([math.sin(theta),math.cos(theta),w])
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
        
        action = RL.choose_action(observation)
        observation_, reward, done = do_action(action,print_this_episode)     # reward = -1 in all cases

        RL.store_transition(observation, action, reward)

        if done:
            # calculate running reward
            ep_rs_sum = sum(RL.ep_rs)
            
            #if 'running_reward' not in globals():
            #    running_reward = ep_rs_sum
            #else:
            #    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            #if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering

            print("episode:", episode, "  reward:", int(ep_rs_sum))

            vt = RL.learn()  # train
            
            break
            
        observation = observation_
    count=count+1
    
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
    no_change=0
    last_action=-1
    print('max_x_num: '+ str(max_x_num))
    print('max_x_val: '+str(max_x_val))

    
print('change print_this_episode_number to this number to print the max_x_num: '+ str(max_x_num))
print(max_x_val)
