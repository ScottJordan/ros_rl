import numpy as np
import rospy
from ros_rl.msg import EnvAct, EnvObs, EnvDescMsg
from ros_rl.srv import GetEnvDesc, GetEnvDescResponse
from std_srvs.srv import Empty, EmptyResponse

from ros_rl.utils.thing import ThingDesc, ThingfromDesc, ThingDescfromMsg, ThingfromMsg

INACTIVE = 0
ACTIVE = 1
FINISHED = 2
state_map = {INACTIVE:'INACTIVE', ACTIVE:'ACTIVE', FINISHED:'FINISHED'}

class EnvDesc(object):
    def __init__(self):
        # Episodic or continuing?
        self.episodic = None

        # If you are going to cut episodes, when should you cut them?
        self.suggestedMaxT = 1

        # How many episodes should an agent be given to learn on this env?
        self.suggestedMaxEps = 1

        # Reward discount parameter. Only used if episodic (not for continuing tasks)
        self.gamma = 1.

        # Description of the actions
        self.actDesc = ThingDesc()

        # Description of the observations
        self.obsDesc = ThingDesc()

        self.minReward = -float("inf")  # Set to -INF if not known or not bounded
        self.maxReward =  float("inf")  # Set to INF if not known or not bounded
        self.minReturn = -float("inf")  # Set to -INF if not known or not bounded
        self.maxReturn =  float("inf")  # Set to INF if not known or not bounded

        # Recommended plot y-axis
        self.suggestedPlotMinPerformance = 0.
        self.suggestedPlotMaxPerformance = 1.

    def toMsg(self):
        msg = EnvDescMsg()
        msg.episodic = self.episodic
        msg.suggestedMaxT = self.suggestedMaxT
        msg.suggestedMaxEps = self.suggestedMaxEps
        msg.gamma = self.gamma
        msg.actDesc = self.actDesc.toMsg()
        msg.obsDesc = self.obsDesc.toMsg()
        msg.minReward = self.minReward
        msg.maxReward = self.maxReward
        msg.minReturn = self.minReturn
        msg.maxReturn = self.maxReturn
        msg.suggestedPlotMinPerformance = self.suggestedPlotMinPerformance
        msg.suggestedPlotMaxPerformance = self.suggestedPlotMaxPerformance
        return msg

def EnvDescfromMsg(msg):
    desc = EnvDesc()
    desc.episodic = msg.episodic
    desc.suggestedMaxT = msg.suggestedMaxT
    desc.suggestedMaxEps = msg.suggestedMaxEps
    desc.gamma = msg.gamma
    desc.actDesc = ThingDescfromMsg(msg.actDesc)
    desc.obsDesc = ThingDescfromMsg(msg.obsDesc)
    desc.minReward = msg.minReward
    desc.maxReward = msg.maxReward
    desc.minReturn = msg.minReturn
    desc.maxReturn = msg.maxReturn
    desc.suggestedPlotMinPerformance = msg.suggestedPlotMinPerformance
    desc.suggestedPlotMaxPerformance = msg.suggestedPlotMaxPerformance
    return desc

class RosEnv(object):
    def __init__(self, env_name, desc, rate=100, act_wait=0.005):
        self.env_name = env_name
        self.desc = desc
        self.act_wait = act_wait
        # setup obs publisher
        self.obs_pub = rospy.Publisher("/RL/env/"+self.env_name+"/obs", EnvObs, queue_size=1)
        # setup action subscriber
        self.act_sub = rospy.Subscriber("/RL/env/"+self.env_name+"/act", EnvAct, self.act_callback, queue_size=1)

        # setup environment services
        rospy.Service("/RL/env/"+self.env_name+"/start", Empty, self.handle_start_request)
        rospy.Service("/RL/env/"+self.env_name+"/reset", Empty, self.handle_reset_request)
        rospy.Service("/RL/env/"+self.env_name+"/stop", Empty, self.handle_stop_request)
        rospy.Service("/RL/env/"+self.env_name+"/getDesc", GetEnvDesc, self.handle_envDesc_request)

        # initalize obs, act, and reward
        self.obs = ThingfromDesc(self.desc.obsDesc)
        self.action = ThingfromDesc(self.desc.actDesc)
        self.reward = 0
        self.terminal = False

        # set rate for looping between observations
        self.rate = rospy.Rate(rate)  # in hz

        self.start_flag = False
        self.reset_flag = False
        self.stop_flag = False
        self.is_reset = False
        self.new_action = False
        self.act_cmds = 0
        self.state = INACTIVE
        self.time_step = 0

    def act_callback(self, msg):
        act = ThingfromMsg(msg.act)
        self.action = self.constrain_actions(act)
        self.new_action = True

    def handle_envDesc_request(self, req):
        resp = GetEnvDescResponse()
        resp.envDesc = self.desc.toMsg()
        return resp

    def handle_start_request(self, req):
        self.start_flag = True
        return EmptyResponse()

    def handle_stop_request(self, req):
        self.stop_flag = True
        self.stop_controllers()
        self.state = INACTIVE
        return EmptyResponse()

    def handle_reset_request(self, req):
        self.reset_flag = True
        self.stop_controllers()
        self.state = FINISHED
        return EmptyResponse()

    def constrain_actions(self, action):
        if self.desc.actDesc.numDisc > 0:
            action.disc = max(min(action.disc, self.desc.actDesc.numDisc), 0)
        if self.desc.actDesc.contDim > 0:
            action.cont = np.clip(action.cont, self.desc.actDesc.contRange[:, 0], self.desc.actDesc.contRange[:, 1])

        return action

    def publish_obs(self):
        msg = EnvObs()
        msg.stamp = rospy.Time.now()
        msg.obs = self.obs.toMsg()
        msg.reward = self.reward
        msg.terminal = self.terminal

        self.obs_pub.publish(msg)

    def reset(self):
        self.resetWorld()
        self.newEpisode()
        self.is_reset = True

    def resetWorld(self):
        raise NotImplementedError

    def inTerminalState(self):
        raise NotImplementedError

    def newEpisode(self):
        raise NotImplementedError

    def send_action(self):
        raise NotImplementedError

    def stop_controllers(self):
        raise NotImplementedError

    def compute_obs(self):
        raise NotImplementedError

    def compute_reward(self):
        raise NotImplementedError

    def run(self):
        # print('state {0}\t start {1}\t reset {2}\t reset_flag {3}\t action {4}'.format(state_map[self.state], self.start_flag, self.is_reset, self.reset_flag, self.new_action))
        if self.state == INACTIVE:
            if self.start_flag:
                if self.is_reset:
                    self.start_flag = False
                    self.state = ACTIVE

                else:
                    self.reset_flag = True
            if self.reset_flag:
                self.reset()
                self.reset_flag = False

        elif self.state == ACTIVE:
            self.is_reset = False
            self.compute_obs()
            self.inTerminalState()
            self.compute_reward()
            self.publish_obs()
            if self.terminal:
                self.stop_controllers()
                self.state = FINISHED
            else:
                rospy.sleep(self.act_wait)
                self.send_action()
        elif self.state == FINISHED:
            print("received {0:d}/{1:d} actions".format(self.act_cmds, self.time_step-1))
            self.reset()
            self.state = INACTIVE
        else:
            print('unknown state')
            self.stop_controllers()
            self.state = INACTIVE
