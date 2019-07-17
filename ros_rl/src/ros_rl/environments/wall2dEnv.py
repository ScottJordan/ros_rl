#! /usr/bin/env python
# Don't forget to make the file executable for rus run to find it

from __future__ import print_function, division

import argparse
import numpy as np
import rospy

from ros_rl.environments.environment import RosEnv, EnvDesc
from ros_rl.utils.thing import ThingDesc, ThingfromDesc


def getEnvConfig():
    desc = EnvDesc()
    desc.episodic = True
    desc.suggestedMaxT = 200
    desc.suggestedMaxEps = 150
    desc.gamma = 1

    world_dim = 2
    act_high = np.ones(world_dim, dtype=np.float64)
    act_range = np.array([-act_high, act_high]).T
    desc.actDesc = ThingDesc(numDisc=0, contRange=act_range)

    obs_high = 2*np.ones(world_dim, dtype=np.float64)
    obs_range = np.array([-obs_high, obs_high]).T
    desc.obsDesc = ThingDesc(numDisc=0, contRange=obs_range)

    desc.minReward = -1.
    desc.maxReward = 10.
    desc.maxReturn = 10
    desc.minReturn = -10

    desc.suggestedPlotMinPerformance = -10
    desc.suggestedPlotMaxPerformance = 10

    return desc


class Wall2dEnv(RosEnv):
    def __init__(self, shape_reward=False):
        super(Wall2dEnv, self).__init__('Wall2d', getEnvConfig(), 175, 0.005)
        self.action_scale = 0.1
        self.shape_reward = shape_reward
        self.goal_distance = 0.05
        self.prev_dist = 1.
        self.goal = np.array([1,0])
        #self.time_step = 0

    def newEpisode(self):
        self.obs = ThingfromDesc(self.desc.obsDesc, random=False)
        self.obs.cont = np.array([-1, 0], dtype=np.float64)
        self.action = ThingfromDesc(self.desc.actDesc)
        self.prev_dist = 1.
        self.reward = 0
        self.terminal = False
        self.time_step = 0
        self.act_cmds = 0

    def inTerminalState(self):
        dist_cond = np.linalg.norm(self.obs.cont-self.goal) < self.goal_distance  # 0.01
        time_cond = self.time_step > self.desc.suggestedMaxT
        self.terminal = dist_cond or time_cond
        return self.terminal

    def compute_obs(self):
        cur_loc = self.obs.cont
        move = self.action.cont * self.action_scale
        new_loc = move + cur_loc
        # if close enough to cross
        if abs(cur_loc[0]) <= 0.1:
            # crossed middle
            if new_loc[0]*cur_loc[0] <= 0:
                # intersected line segment
                wall_width = 0.3
                if ((cur_loc[1] >= -wall_width and cur_loc[1] <= wall_width) and (new_loc[1] >= -wall_width and new_loc[1] <= wall_width)):
                    new_x = np.sign(cur_loc[0]) * 0.001
                    new_y = new_loc[1]#((new_x-cur_loc[0])/move[0])*move[1] + cur_loc[1]
                    new_loc = np.array([new_x, new_y])

        self.obs.cont = new_loc
        self.obs.cont = np.clip(self.obs.cont, self.desc.obsDesc.contRange[:, 0], self.desc.obsDesc.contRange[:, 1])
        self.time_step += 1

    def compute_reward(self):
        dist = np.linalg.norm(self.obs.cont-self.goal, ord=1)
        self.reward = -0.05#0.
        if self.shape_reward:
            self.reward = (self.prev_dist - dist) * 1.
        if np.linalg.norm(self.obs.cont-self.goal) < self.goal_distance:
            self.reward += 10.0
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
    args = parser.parse_args()
    rospy.init_node('Wall2dEnv', anonymous=True)

    env = Wall2dEnv(shape_reward=args.shape)
    print('Env Created')

    # read and publish the incoming data
    while not rospy.is_shutdown():
        res = env.run()
        if res == -1:
            break
        env.rate.sleep()
