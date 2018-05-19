import numpy as np

from ros_rl.msg import ThingMsg, ThingDescMsg

class Thing(object):
    def __init__(self, disc=None, cont=None):
        if disc is not None:
            self.disc = disc
        else:
            self.disc = -1

        if cont is not None:
            self.cont = np.array(cont)
        else:
            self.cont = np.array([])

    def toMsg(self):
        msg = ThingMsg()
        msg.disc = self.disc
        if self.cont.shape[0] > 0:
            msg.contDim = self.cont.shape[0]
            msg.cont = self.cont.tolist()
        else:
            msg.contDim = 0
            msg.cont = []
        return msg

def ThingfromMsg(msg):
    thing = Thing()
    thing.disc = msg.disc
    if msg.contDim > 0:
        thing.cont = np.array(msg.cont)
    return thing

def ThingfromDesc(desc, random=False):
    thing = Thing()
    if desc.numDisc > 0:
        if random:
            thing.disc = int(np.random.choice(list(range(desc.numDisc)), size=1))
        else:
            thing.disc = 0
    if desc.contDim > 0:
        if random:
            crange = (desc.contRange[:, 1] - desc.contRange[:, 0])
            thing.cont = np.random.rand(desc.contDim) * crange + desc.contRange[:, 0]
        else:
            thing.cont = np.zeros(desc.contDim, dtype=np.float64)
    return thing

class ThingDesc(object):
    def __init__(self, numDisc=0, contRange=None):
        self.numDisc = numDisc
        if contRange is not None:
            self.contRange = np.array(contRange)
            self.contDim = self.contRange.shape[0]
        else:
            self.contDim = 0
            self.contRange = np.array([])

    def toMsg(self):
        msg = ThingDescMsg()
        msg.numDisc = self.numDisc
        msg.contDim = self.contDim
        if self.contDim > 0:
            msg.contRangeLow = self.contRange[:, 0].tolist()
            msg.contRangeHigh = self.contRange[:, 1].tolist()
        else:
            msg.contRangeLow = []
            msg.contRangeHigh = []

        return msg


def ThingDescfromMsg(msg):
    desc = ThingDesc()
    desc.numDisc = msg.numDisc
    desc.contDim = msg.contDim
    if desc.contDim > 0:
        desc.contRange = np.array([msg.contRangeLow, msg.contRangeHigh]).T
    return desc
