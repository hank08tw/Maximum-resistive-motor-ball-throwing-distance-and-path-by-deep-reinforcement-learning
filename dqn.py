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
print_this_episode_number=5000#1423
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
energy_diff_coefficient=10000#1.1 (10 0.38) (1 0.36)  (0.1 0.33)          
#算法跑幾個episode
num_episode=5000
# reproducible
np.random.seed(1)
tf.set_random_seed(1)

#用來畫圖 儲存每個時間點theta跟w的值
theta_box = []
w_box = []
alpha_box= []
moment=[]

action_space=['forward','dormant','backward','throw']
image = svgwrite.Drawing('dqn_animation.svg', size=(1024,1024))
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




# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net -----feature-------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]
        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]
        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()










#from maze_env import Maze
#from RL_brain import DeepQNetwork


def throw_ball(RL):
    global L_m,L,theta,w,alpha,g,delta_t,R,m,k,i,max_x_num,max_x_val,action_space,count,num_episode
    step = 0
    print_this_episode=False
    for episode in range(num_episode):
        # initial observation
        #observation = env.reset()
        print('episode: '+str(episode))
        if episode==print_this_episode_number:
            print_this_episode=True
        else:
            print_this_episode=False
        observation=np.array([math.sin(theta),math.cos(theta),w])
        while True:
            # fresh env
            #env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)
            #print('action: '+str(action))
            # RL take action and get next observation and reward
            observation_, reward, done = do_action(action,print_this_episode)
            #print('after do action')
            RL.store_transition(observation, action, reward, observation_)
            #print('after transition')
            if (step > 200) and (step % 5 == 0):
                RL.learn()
                #print('after learn')

            # swap observation
            observation = observation_
            #print('after swap observation')

            # break while loop when end of this episode
            if done:
                break
            step += 1
            #print('after step')
        theta=0
        w=0
        alpha=0
        #print('episode: '+str(episode))
        count=count+1
        print('max_num: '+str(max_x_num))
        print('max_val: '+str(max_x_val))
    # end of game
    print('game over')
    #env.destroy()


if __name__ == "__main__":
    # maze game
    #env = Maze()
    RL = DeepQNetwork(len(action_space), 3,
                      learning_rate=0.01,
                      reward_decay=0.9,#0.9
                      e_greedy=0.9,
                      replace_target_iter=100,
                      memory_size=2000,
                      # output_graph=True
                      )
    title('x axis is theta, y axis is w')
    throw_ball(RL)
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
    #env.after(100, throw_ball)
    #env.mainloop()
    print('change print_this_episode_number to this number to print the max_x_num: '+ str(max_x_num))
    print(max_x_val)
    image.save()
    """
    print_this_episode_number=max_x_num


    RL = DeepQNetwork(len(action_space), 2,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    #throw_ball(RL2)
    #env.after(100, throw_ball)
    #env.mainloop()
    print(max_x_num)
    print(max_x_val)
    image.save()
    """

    
RL.plot_cost()