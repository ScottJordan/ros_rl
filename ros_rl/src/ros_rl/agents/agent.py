from __future__ import print_function, division

import numpy as np
import rospy
from ros_rl.msg import EnvAct, EnvObs
from ros_rl.srv import GetEnvDesc, GetEnvDescRequest
from rospy.exceptions import ROSException, ROSInterruptException
from std_srvs.srv import Empty, EmptyRequest

from ros_rl.environments.environment import EnvDescfromMsg
from ros_rl.utils.thing import Thing, ThingfromMsg


class RosAgent(object):
    def __init__(self, env_id, agent_fn):
        self.env_id = env_id

        self.obs_sub = rospy.Subscriber("/RL/env/"+env_id+"/obs", EnvObs, self.obs_callback, queue_size=1)
        self.act_pub = rospy.Publisher("/RL/env/"+env_id+"/act", EnvAct, queue_size=1)
        desc_service = "/RL/env/"+env_id+"/getDesc"
        try:
            rospy.wait_for_service(desc_service, timeout=10.)
            getEnvDesc = rospy.ServiceProxy(desc_service, GetEnvDesc)
            req = GetEnvDescRequest()
            edm = getEnvDesc(req)
            self.envDesc = EnvDescfromMsg(edm.envDesc)

        except ROSInterruptException as e:
            print('ROS Interrupt Exception...Exiting')
            exit(1)
        except ROSException as e:
            print('Wait for service timed out')
            print(str(e))
            exit(1)
        except rospy.ServiceException as e:
            print("Service did not process request: " + str(e))
            exit(1)
        except Exception as e:
            print("Some other exception happened")
            print(str(e))
            exit(1)

        self.agent = agent_fn(self.envDesc)
        self.episode = 0
        self.env_active = False
        self.new_obs_flag = False
        self.cur_obs = None
        self.act = Thing()
        self.new_obs = None
        self.reward = 0
        self.terminal = False
        self.first_step = True
        self.reward_hist = []
        self.returns = []

    def obs_callback(self, msg):
        self.new_obs = ThingfromMsg(msg.obs)
        self.reward = msg.reward
        self.terminal = msg.terminal
        self.new_obs_flag = True
        #print(self.new_obs.cont, self.reward, self.terminal, self.new_obs_flag)

    def send_start(self):
        try:
            service_name = "/RL/env/"+self.env_id+"/start"
            rospy.wait_for_service(service_name, timeout=10.)
            start_service = rospy.ServiceProxy(service_name, Empty)
            req = EmptyRequest()
            start_service(req)

        except ROSInterruptException as e:
            print('ROS Interrupt Exception...Exiting')
            print(str(e))
            exit(1)
        except ROSException as e:
            print('Wait for service timed out')
            print(str(e))
            exit(1)
        except rospy.ServiceException as e:
            print("Service did not process request: " + str(e))
            exit(1)
        except Exception as e:
            print("Some other exception happened")
            print(str(e))
            exit(1)
        self.env_active = True

    def constrain_actions(self, action):
        if self.envDesc.actDesc.numDisc > 0:
            action.disc = max(min(action.disc, self.envDesc.actDesc.numDisc), 0)
        if self.envDesc.actDesc.contDim > 0:
            action.cont = np.clip(action.cont, self.envDesc.actDesc.contRange[:, 0], self.envDesc.actDesc.contRange[:, 1])

        return action

    def publish_action(self):
        envAct = EnvAct()
        envAct.stamp = rospy.Time.now()
        envAct.act = self.act.toMsg()
        self.act_pub.publish(envAct)

    def run(self, max_eps=None):
        if not max_eps:
            max_eps = self.envDesc.suggestedMaxEps
        if self.episode < max_eps:
            if not self.env_active:
                self.reward_hist.append([])
                self.send_start()
            else:
                if self.new_obs_flag:
                    if self.first_step:
                        self.first_step = False
                    else:
                        self.reward_hist[-1].append(self.reward)
                        self.agent.update(self.cur_obs, self.act, self.reward, self.new_obs, self.terminal)
                    self.cur_obs = self.new_obs
                    self.new_obs_flag = False

                    if self.terminal:
                        self.env_active = False
                        self.episode += 1
                        ret = np.sum(self.reward_hist[-1])
                        self.returns.append(ret)
                        self.agent.new_episode()
                        self.first_step = True
                        print("episode: {0:d} {1:.3f}".format(self.episode, ret), len(self.reward_hist[-1]))
                    else:
                        act = self.agent.get_action(self.cur_obs)
                        self.act = act  # self.constrain_actions(act)
                        self.publish_action()
            return False, None
        else:
            print('{0:d} episodes: average ret: {1:.3f} max ret:{2:.3f}'.format(self.episode, np.mean(self.returns), np.max(self.returns)))
            return True, self.returns#np.mean(self.returns)
