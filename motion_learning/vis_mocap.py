import numpy as np
import math
from utils import bullet_client
from utils.humanoid_kin import HumanoidSkeleton
from utils.humanoid_mocap import HumanoidMocap
from utils.humanoid_vis import HumanoidVis
import pybullet as p1

#from IPython import embed
import time

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
  
from lma_extractor import LMAExtractor
from emotion_classifier import EmotionClassifier


###
ACT_STEPTIME  = 1./30.
SUBSTEPS = 20
VIS_STEP = 1.0/150
SIM_STEPTIME  = ACT_STEPTIME / SUBSTEPS
###

class VisMocapEnv():
    def __init__(self, mocap_file, pybullet_client=None, model="humanoid3d"):
      self._isInitialized = False
      self.rand_state = np.random.RandomState()
      self._motion_file = mocap_file
      self.enable_draw = True
      self.follow_character = True
      self._model = model
      self.init()
      self.has_looped = False
      self._previous_mocap_phase = 0.0
      self._current_mocap_phase = 0.0

    def init(self):
      """
        initialize environment with bullet backend
      """
      # mocap data
      if self._model == "humanoid3d":
        char_file = "data/characters/humanoid3d.txt"
        ctrl_file = "data/controllers/humanoid3d_ctrl.txt"
        self._skeleton = HumanoidSkeleton(char_file, ctrl_file)
        self._mocap = HumanoidMocap(self._skeleton, self._motion_file)
        print("mocap duration: %f" % self._mocap._cycletime)
      elif self._model == "atlas":
        char_file = "data/characters/atlas.txt"
        ctrl_file = "data/controllers/atlas_ctrl.txt"
        self._skeleton = HumanoidSkeleton(char_file, ctrl_file)
        self._mocap = HumanoidMocap(self._skeleton, self._motion_file)
        print("mocap duration: %f" % self._mocap._cycletime)
      elif self._model == "atlas_jason":
        char_file = "data/characters/atlas_jason.txt"
        ctrl_file = "data/controllers/atlas_jason_ctrl.txt"
        self._skeleton = HumanoidSkeleton(char_file, ctrl_file)
        self._mocap = HumanoidMocap(self._skeleton, self._motion_file)
        print("mocap duration: %f" % self._mocap._cycletime)
      else:
        assert(False)

      self._visual = HumanoidVis(self._skeleton, self._model)
      #color = [227/255, 170/255, 14/255, 1] # sim
      color = [44/255, 160/255, 44/255, 1] # ref
      self._char = self._visual.add_character("mocap", color)
      self._visual.camera_follow(self._char, 2, 0, 0)

      self._pybullet_client = self._visual._pybullet_client
      self._play_speed = self._pybullet_client.addUserDebugParameter("play_speed", 0, 2, 1.0)
      self._phase_ctrl = self._pybullet_client.addUserDebugParameter("frame", 0, 1, 0)


    def reset(self, phase=None):
      startTime = self.rand_state.rand() if phase is None else phase
      startTime *= self._mocap._cycletime
      self.t = startTime
      self.start_phase = self._mocap.get_phase(self.t)
      self._previous_mocap_phase = self.start_phase

      count, pose, vel = self._mocap.slerp(startTime)

      if self.enable_draw:
        self.synchronize_sim_char()
        self._prev_clock = time.time()
        self._visual.camera_follow(self._char)

    def step(self):
      speed = self._pybullet_client.readUserDebugParameter(self._play_speed)
      phase = self._pybullet_client.readUserDebugParameter(self._phase_ctrl)
      #phase /= self._mocap.num_frames
      
      if (self.start_phase != phase):
        self.reset(phase)

      self._current_mocap_phase = self._mocap.get_phase(self.t)
      if(self._previous_mocap_phase > self._current_mocap_phase):
        self.has_looped = True

      self._previous_mocap_phase = self._current_mocap_phase

      self.update(VIS_STEP, speed)

    def update(self, timeStep, speed):
      self.wait_till_timestep(VIS_STEP)
      self.synchronize_sim_char()
      if self.follow_character:
        self._visual.camera_follow(self._char)
        #cnt, pos = self._mocap.get_com_pos(self.t, True)
        #pos += cnt * self._mocap._cyc_offset
        #pos[1] = 1
        #self._visual.camera_follow(self._char, None, None, None, pos)

      self.t += timeStep * speed

    def synchronize_sim_char(self):
      count, pose, vel = self._mocap.slerp(self.t)
      pose[:3] += count * self._mocap._cyc_offset
      self._visual.set_pose(self._char, pose, vel)

    def getKeyboardEvents(self):
      return self._pybullet_client.getKeyboardEvents()

    def isKeyTriggered(self, keys, key):
      o = ord(key)
      #print("ord=",o)
      if o in keys:
        return keys[ord(key)] & self._pybullet_client.KEY_WAS_TRIGGERED
      return False

    def wait_till_timestep(self, timeStep):
      time_remain = timeStep - (time.time() - self._prev_clock)
      if time_remain > 0:
        time.sleep(time_remain)
      self._prev_clock = time.time()

    def get_pose_and_links(self):
      return self._get_full_model_pose(self._visual.characters["mocap"])
    
    def _get_full_model_pose(self, phys_model):
      """ Get current pose and velocity expressed in general coordinate
        Unlike _get_model_pose it also returns the Position and Velocity of each link/joint.
        Inputs:
          phys_model

        Outputs:
          pose
          vel
      """
      pose = []
      vel = []

      links_pos = {}
      links_orn = {}

      vel_dict = {}

      # root position/orientation and vel/angvel
      pos, orn = self._pybullet_client.getBasePositionAndOrientation(self._visual.characters["mocap"])
      linvel, angvel = self._pybullet_client.getBaseVelocity(self._visual.characters["mocap"])
      pose += pos
      if orn[3] < 0:
        orn = [-orn[0], -orn[1], -orn[2], -orn[3]]
      pose.append(orn[3])  # w
      pose += orn[:3] # x, y, z
      vel += linvel
      vel += angvel

      vel_dict["root"] = [linvel, angvel]

      for i in range(self._skeleton.num_joints):
        j_info = self._pybullet_client.getJointStateMultiDof(phys_model, i)
        orn = j_info[0]
        if len(orn) == 4:
          pose.append(orn[3])  # w
          pose += orn[:3] # x, y, z
        else:
          pose += orn
        vel += j_info[1]

        l_info = self._pybullet_client.getJointInfo(phys_model, i)
        
        if(not l_info[12].decode('UTF-8') == "root"):
          vel_dict[l_info[12].decode('UTF-8')] = j_info[1]
        
        link_name = l_info[12].decode('UTF-8')
        
        l_info = self._pybullet_client.getLinkState(phys_model, i)
        
        links_pos[link_name] = l_info[0]
        links_orn[link_name] = l_info[1]

      pose = np.array(pose)
      vel = self._skeleton.padVel(vel)
      
      return pose, vel, links_pos, links_orn, vel_dict


def show_mocap(mocap_file, model, record_lma='', predict_emotion=True):
  env = VisMocapEnv(mocap_file, None, model)
  #env._mocap.show_com()
  env.reset()
  if(record_lma != ""):
    lma_extractor = LMAExtractor(env, record_lma, append_to_file=True)
  else:
    lma_extractor = LMAExtractor(env, append_to_file=False, label="NONE")

  if(predict_emotion):
    emotion_predictor = EmotionClassifier()

  while True:
    # LMA Features

    if(not env.has_looped):
      lma_extractor.record_frame()
    else:
      if(len(lma_extractor.get_lma_features()) >= 1):
        print(emotion_predictor.predict_emotion_coordinates(lma_extractor.get_lma_features()))
        lma_extractor.clear()

      print(emotion_predictor.predict_final_emotion())
      break

    # Every 5 LMA features, run predictor
    if(len(lma_extractor.get_lma_features()) >= 10):
      print(emotion_predictor.predict_emotion_coordinates(lma_extractor.get_lma_features()))
      lma_extractor.clear()
    
    env.step()


if __name__=="__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--mocap", type=str, default='data/motions/humanoid3d_jump.txt', help="task to perform")
  parser.add_argument("--model", type=str, default='humanoid3d', help="model")
  parser.add_argument("--record_lma",default='', action="store_true", help="specify a file name if you want to store the lma features on a file")

  parser.add_argument("--predict_emotion",default=True, action="store_true" , help="specify whether you want to output the predicted emotional coordinates")

  args = parser.parse_args()

  show_mocap(args.mocap, args.model, args.record_lma, args.predict_emotion)
