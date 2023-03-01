#!/usr/bin/env python

from carla_env.carla_env import CarlaEnv

import numpy as np

import rospy
from geometry_msgs.msg import Twist

from datetime import datetime


import signal
import time
import os
import pickle

action = [0., 0., 0.]

def handler(signum, frame):
    msg = "Ctrl-c was pressed. Exiting now"
    exit(1)

signal.signal(signal.SIGINT, handler)

def teleop_callbak(data):
    steer = -data.angular.z
    throttle = data.linear.x
    brake = 0 

    if throttle < 0:
        brake = -throttle
        throttle = 0

    steer = np.clip(steer / 30, -1, 1) 
    global action
    action = [throttle, brake, steer]

def run():
    print("setting up Carla-Gym")
    env = CarlaEnv()

    print("Starting loop")
    obs = env.reset()
    i = 0
    
    times = []
    while not rospy.is_shutdown():
        print("Loop - ", i)

        global action
        obs, reward, done, state, act = env.step(action)
        env.render()

        i+=1 
        if done: break


if __name__=='__main__':
    rospy.init_node('carla', anonymous=True)
    rospy.Subscriber("/cmd_vel", Twist, teleop_callbak)
    run()