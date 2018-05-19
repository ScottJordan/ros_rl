#! /usr/bin/env python
# Don't forget to make the file executable for rus run to find it

from __future__ import print_function, division

import argparse
import numpy as np
import rospy

from ros_rl.src.ros_rl.environments.environment import RosEnv, EnvDesc
from ros_rl.src.ros_rl.utils.thing import ThingDesc, ThingfromDesc


def getEnvConfig():
    desc = EnvDesc()
    desc.episodic = True
    desc.suggestedMaxT = 50
    desc.suggestedMaxEps = 50
    desc.gamma = 1

    world_dim = 3
    act_high = np.ones(world_dim, dtype=np.float64)
    act_range = np.array([-act_high, act_high]).T
    desc.actDesc = ThingDesc(numDisc=0, contRange=act_range)

    obs_high = np.ones(world_dim, dtype=np.float64)
    obs_range = np.array([-obs_high, obs_high]).T
    desc.obsDesc = ThingDesc(numDisc=0, contRange=obs_range)

    desc.minReward = -1.
    desc.maxReward = 1.
    desc.maxReturn = 10
    desc.minReturn = -10

    desc.suggestedPlotMinPerformance = -10
    desc.suggestedPlotMaxPerformance = 10

    return desc


class SimpleEnv(RosEnv):
    def __init__(self, name='Simple', shape_reward=False):
        super(SimpleEnv, self).__init__(name, getEnvConfig(), 175, 0.005)
        self.name = name
        self.action_scale = 0.1
        self.shape_reward = shape_reward
        self.goal_distance = 0.02
        self.prev_dist = 1.
        #self.time_step = 0

    def newEpisode(self):
        self.obs = ThingfromDesc(self.desc.obsDesc, random=False)
        self.obs.cont = np.copy(self.desc.obsDesc.contRange[:, 0])
        self.action = ThingfromDesc(self.desc.actDesc)
        self.prev_dist = 1.
        self.reward = 0
        self.terminal = False
        self.time_step = 0
        self.act_cmds = 0

    def inTerminalState(self):
        dist_cond = np.linalg.norm(self.obs.cont) < self.goal_distance # 0.01
        time_cond = self.time_step > self.desc.suggestedMaxT
        self.terminal = dist_cond or time_cond
        return self.terminal

    def compute_obs(self):
        self.obs.cont += self.action.cont * self.action_scale
        self.time_step += 1

    def compute_reward(self):
        dist = np.linalg.norm(self.obs.cont)
        self.reward = -.1#0
        if self.shape_reward:
            self.reward += (self.prev_dist - dist) * 1

        if np.linalg.norm(self.obs.cont) < self.goal_distance:
            self.reward += 1.
        # else:
        #     self.reward += 0.

        self.prev_dist = dist

    def stop_controllers(self):
        self.action = ThingfromDesc(self.desc.actDesc)

    def send_action(self):
        if self.new_action:
            self.act_cmds += 1
        self.new_action = False
        pass  # no need to send anything. We are not using outside controllers

    def resetWorld(self):
        pass  # no need to reset anything self.obs contains the information and not using gazebo or the robot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape', action='store_true')
    parser.add_argument('--name', default='1', type=str)
    args = parser.parse_args()
    name = 'Simple_'+args.name
    rospy.init_node(name, anonymous=True)

    env = SimpleEnv(name=name, shape_reward=args.shape)
    print('Env Created', name)

    # read and publish the incoming data
    while not rospy.is_shutdown():
        res = env.run()
        if res == -1:
            break
        env.rate.sleep()
