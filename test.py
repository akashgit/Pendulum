import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
slim = tf.contrib.slim
from keras import metrics
'''Simple Pendulum'''
import math
from scipy.integrate import solve_ivp
import random
from tqdm import tqdm


LENGTH=np.arange(1,4)
MASS=np.arange(1,2)
B=[0.1,0.2,0.3,0.4,0.5,0]
BATCH_SIZE=10
OUT=30


def generateData(L,m,b):
    st = 0
    et = OUT
    ts = 1
    g = 9.8


    def sim_pen_eq(t, theta):
        dtheta2_dt = (-b / m) * theta[1] + (-g / L) * np.sin(theta[0])
        dtheta1_dt = theta[1]
        return [dtheta1_dt, dtheta2_dt]

    def sim():
        theta1_ini = 0
        theta2_ini = 3
        theta_ini = [theta1_ini, theta2_ini]
        t_span = [st, et + ts]
        t = np.arange(st, et + ts, ts)
        sim_points = len(t)
        l = np.arange(0, sim_points, 1)

        theta12 = solve_ivp(sim_pen_eq, t_span, theta_ini, t_eval=t)
        theta1 = theta12.y[0, :]
        theta2 = theta12.y[0, :]
        return theta1

    return sim()

def batch(l=[1],m=[1],b=[0]):
    X = []
    L = []
    M = []
    B = []
    for i in range(BATCH_SIZE):
        l_ = random.choice(l)
        m_ = random.choice(m)
        b_ = random.choice(b)
        L.append(l_)
        M.append(m_)
        B.append(b_)
        X.append(generateData(l_,m_,b_)[:-1])
    return np.asarray(X),np.asarray(L),np.asarray(M),np.asarray(B),




def simple_learned_sim(l, h_layer=100, out=OUT, reuse=False):
    with tf.variable_scope("simple") as scope:
        if reuse:
            scope.reuse_variables()
        l = l*tf.ones([BATCH_SIZE,1])
        t = slim.fully_connected(l,h_layer, activation_fn = tf.nn.relu)
        t = slim.fully_connected(l, h_layer//2, activation_fn=tf.nn.relu)
        t = slim.fully_connected(t,out, activation_fn = None)
        return t

def sim_with_friction(t, c, h_layer=100, out=OUT, reuse=False):
    with tf.variable_scope("with_friction") as scope:
        if reuse:
            scope.reuse_variables()
        t = slim.fully_connected(t, out, activation_fn=tf.nn.relu)
        c = slim.fully_connected(c, out, activation_fn=tf.nn.relu)

        tc = tf.concat([t,c],1)

        t = slim.fully_connected(tc, h_layer, activation_fn=tf.nn.relu)
        t = slim.fully_connected(t, h_layer, activation_fn=tf.nn.relu)
        t = slim.fully_connected(t, out, activation_fn=None)
        return t

def train():
    tf.reset_default_graph()

    t_input, l_input1, _, _ = batch(LENGTH,MASS,[0.0])
    t_bar = simple_learned_sim(l_input1)

    simple_loss = tf.reduce_mean(metrics.mean_squared_error(t_input, t_bar))

    x_input, l_input2, _, _ = batch(LENGTH,MASS,B)
    c = tf.random_normal((BATCH_SIZE, 2), 0, 1, dtype=tf.float32)
    t_bar_sim = simple_learned_sim(l_input2,reuse=True)
    x_bar = sim_with_friction(t_bar_sim, c)

    friction_loss = tf.reduce_mean(metrics.mean_squared_error(x_input, x_bar))


    t_vars = tf.trainable_variables()

    s_vars = [var for var in t_vars if 'simple' in var.name]
    f_vars = [var for var in t_vars if 'with_friction' in var.name]

    # Use ADAM optimizer
    s_optim = tf.train.AdamOptimizer().minimize(simple_loss, var_list=s_vars)
    f_optim = tf.train.AdamOptimizer().minimize(friction_loss, var_list=f_vars)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    loss_s=[]
    loss_f=[]

    for epoch in tqdm(range(100000)):

        s,_,t_out = sess.run([simple_loss,s_optim,t_bar],feed_dict={})
        loss_s.append(s)

    plt.figure()
    # for i in range(len(t_input)):
    plt.plot(t_input[0])
    plt.plot(t_out[0],'r-')
    plt.show()

    for epoch in tqdm(range(100000)):

        f,_,x_out = sess.run([friction_loss, f_optim,x_bar],feed_dict={})
        loss_f.append(f)

    plt.figure()
    plt.plot(x_input[0])
    plt.plot(x_out[0],'r-')
    plt.show()

    plt.figure()
    plt.plot(loss_f)
    plt.plot(loss_s)
    plt.show()

train()