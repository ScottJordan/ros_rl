#! /usr/bin/env python
from __future__ import print_function, division

import argparse
import matplotlib.pyplot as plt
import numpy as np
import rospy

from ros_rl.src.ros_rl.agents.agent import RosAgent
from ros_rl.src.ros_rl.utils.thing import ThingfromDesc

class RandomAgent(object):
    def __init__(self, envDesc):
        self.envDesc = envDesc

    def get_action(self, obs):
        act = ThingfromDesc(self.envDesc.actDesc, random=True)
        return act

    def update(self, obs, act, reward, next_obs, terminal):
        pass

    def new_episode(self):
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--num_trials', type=int, default=5)
    # parser.add_argument('--logdir', type=str, default='./')
    args = parser.parse_args()
    rospy.init_node('RandomAgent', disable_signals=True)
    print('You are running a random agent on environment: ', args.env)
    agent_fn = lambda desc: RandomAgent(desc)

    results = []
    for trial in range(args.num_trials):
        agent = RosAgent(args.env, agent_fn)
        print('trial', trial)
        while not rospy.is_shutdown():
            done, res = agent.run()
            if done:
                results.append(res)
                break
            rospy.sleep(0.001)

    mean_ret = np.mean(results, axis=0)
    std_ret = np.std(results, axis=0)
    plt.errorbar(x=list(range(len(results[0]))),y=mean_ret, yerr=std_ret)
    plt.show()



if __name__ == '__main__':
    main()
