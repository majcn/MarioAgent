import math, random, sys, time

from rlglue.agent.Agent import Agent
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.utils.TaskSpecVRLGLUE3 import TaskSpecParser
from collections import defaultdict

FAST_MODE = True

class Monster():
    TYPES_TO_NAMES = ["Mario", "Red Koopa", "Green Koopa", "Goomba",
                      "Spikey", "Piranha Plant", "Mushroom", "Fire Flower",
                      "Fireball", "Shell", "Big Mario", "Fiery Mario"]
    def __init__(self, x, y, sx, sy, m_type, winged):
        self.x = x
        self.y = y
        self.sx = sx
        self.sy = sy
        self.m_type = m_type
        self.m_name = Monster.TYPES_TO_NAMES[m_type]
        self.winged = winged
        self.canJumpOn = 1 if m_type not in [4] else 0

def get_monsters(observation):
    monsters = []
    for i in range(len(observation.intArray[1:])/2):
        m_type = observation.intArray[1 + 2*i]
        winged = observation.intArray[2 + 2*i]
        x, y, sx, sy = observation.doubleArray[4*i : 4*(i+1)]
        monsters.append(Monster(x, y, sx, sy, m_type, winged))
    return monsters

def get_mario(monsters):
    for monster in monsters:
        if monster.m_type in [0, 10, 11]:
            return monster
    else:
        return None
    
def get_world(observation):
    monsters = get_monsters(observation)
    mario = get_mario(monsters)
    mPos = list(observation.charArray).index("M")
    w = ""
    for i in range(-4, 5):
        t_pos = i*22+mPos
        if t_pos < 5 or t_pos > 347:
            w += "     "
        else:
            w += "".join(observation.charArray[t_pos:t_pos+5])
        w += '\n'
    ms = sorted([(int(m.x-mario.x), int(m.y-mario.y), m.canJumpOn, m.winged) for m in monsters if m != mario])
    w += "".join([str(i[0]) + "," + str(i[1]) + "/" + str(i[2]) + str(i[3]) + "#" for i in ms if i[0] <= 3 and i[0] >= 0])

    if mario.sy > 0.001:
        dy = "INC"
    elif mario.sy < 0.001:
        dy = "DEC"
    else:
        dy = "STD"
    isMonsterNear = any([i[0] <= 3 and i[0] >= 0 for i in ms])
    wof = (int(mario.x), int(mario.y), dy, isMonsterNear)
    return w, wof, mario
    
def argmax(lst):
    return lst.index(max(lst))

class MajcnAgent(Agent):
    def agent_init(self, taskSpecString):
        if taskSpecString.find("Mario-v1") != -1:
            print "Task specification contains Mario-v1"
        else:
            print "Task specification does not contain string Mario-v1"
            exit()
            
        self.ACTIONS = [[1, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0], [-1, 0, 0], [-1, 1, 0]]

        self.nr_trials = 0
        self.total_steps = 0
        self.total_finished = 0
        self.step_number = 0
        self.reward = 0.
        self.f = open("output.out", "w")
        
        #self.cur_actions
        #self.cur_states
        #self.cur_statesW
        #self.cur_rewards
        self.q = {}
        self.visits = defaultdict(int)
        
        self.maxX = 0
        
    def getActionCode(self, intArray):
        return self.ACTIONS.index(intArray)
    
    def getActionArray(self, code):
        return self.ACTIONS[code]
    
    def arrayToAction(self, intArray):
        action = Action(3, 0)
        action.intArray = intArray
        return action

    def agent_cleanup(self):
        pass
    
    def agent_freeze(self):
        pass
    
    def agent_message(self,inMessage):
        return None
        
    def agent_start(self, observation):
        self.reward = 0.
        self.step_number = 0
        self.nr_trials += 1
        self.cur_actions = []
        self.cur_states = []
        self.cur_statesW = []
        self.cur_rewards = []
        return self.get_action(observation)
    
    def fixQ(self, reward, end=False):
        self.cur_rewards.append(reward + (0.01 if self.mario.sx > 1. else 0.))
        if len(self.cur_states) < 2:
            return
        cs = self.cur_states[-2]
        ca = self.cur_actions[-2]
        ns = self.cur_states[-1]
        self.visits[(cs, ca)] += 1.
        alpha_n = 1./self.visits[(cs, ca)]
        self.q[cs][ca] = (1.-alpha_n) * self.q[cs][ca] + alpha_n * (self.cur_rewards[-2] + self.cur_rewards[-1] + max(self.q[ns]) - self.q[cs][ca])
        if cs == ns:
            self.q[cs][ca] -= 5.
        if end:
            cs = self.cur_states[-1]
            ca = self.cur_actions[-1]
            self.visits[(cs, ca)] += 1.
            alpha_n = 1./self.visits[(cs, ca)]
            self.q[cs][ca] = (1.-alpha_n) * self.q[cs][ca] + alpha_n * (self.cur_rewards[-1]*2 - self.q[cs][ca])
            
        cs = self.cur_statesW[-2]
        ca = self.cur_actions[-2]
        ns = self.cur_statesW[-1]
        self.visits[(cs, ca)] += 1.
        alpha_n = 1./self.visits[(cs, ca)]
        self.q[cs][ca] = (1.-alpha_n) * self.q[cs][ca] + alpha_n * (self.cur_rewards[-2] + self.cur_rewards[-1] + max(self.q[ns]) - self.q[cs][ca])
        if end:
            cs = self.cur_statesW[-1]
            ca = self.cur_actions[-1]
            self.visits[(cs, ca)] += 1.
            alpha_n = 1./self.visits[(cs, ca)]
            self.q[cs][ca] = (1.-alpha_n) * self.q[cs][ca] + alpha_n * (self.cur_rewards[-1]*2 - self.q[cs][ca])
    
    def agent_step(self, reward, observation):
        self.reward += reward
        self.step_number += 1
        self.total_steps += 1
        self.fixQ(reward)
        return self.get_action(observation)
    
    def agent_end(self, reward):
        self.fixQ(reward, True)
        self.reward += reward
        print >> self.f, "%f,%f" % (self.reward, self.mario.x)
        print "Trial %d ended after %d steps. On position: %.2f. States = %d" % (self.nr_trials, self.step_number, self.mario.x, len(self.q))
        if reward > 50:
            self.total_finished += 1
            print "FINISHED: %d times" % (self.total_finished)
        if FAST_MODE:
            last_x = self.cur_states[-1][0]
            if last_x > self.maxX:
                self.maxX = last_x
                for i in range(len(self.cur_states)):
                    x,y,dy,cm = self.cur_states[i]
                    a = self.cur_actions[i]
                    if x < last_x - 10:
                        self.q[(x,y,dy,cm)][a] += 10.
            
    
    def get_action(self, observation):
        w, wof, self.mario = get_world(observation)
        
        if wof in self.q:
            action = self.getActionArray(argmax(self.q[wof]))
            if w not in self.q:
                self.q[w] = [0. for _ in self.ACTIONS]
        elif w in self.q:
            action = self.getActionArray(argmax(self.q[w]))
            self.q[wof] = [i for i in self.q[w]]
        else:
            self.q[w] = [0. for _ in self.ACTIONS]
            self.q[wof] = [0. for _ in self.ACTIONS]
            action = self.ACTIONS[0]
        
        self.cur_actions.append(self.getActionCode(action))
        self.cur_states.append(wof)
        self.cur_statesW.append(w)

        return self.arrayToAction(action)

if __name__=="__main__":        
    AgentLoader.loadAgent(MajcnAgent())
