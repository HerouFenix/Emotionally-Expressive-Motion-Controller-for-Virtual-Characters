import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)

sys.path.append(parent)
sys.path.append(os.path.join(parent, "motion_learning"))

import numpy as np
import math
from utils import bullet_client
from utils.humanoid_kin import HumanoidSkeleton
from utils.humanoid_mocap import HumanoidMocap
from utils.humanoid_vis import HumanoidKinNoVis
import pybullet as p1

from utils.humanoid_kin import JointType

#from IPython import embed
import time

# This script extracts all Deepmimic mocap files in the mocap_data directory's LMA features

input_directory = 'kin_mocap_reduced/'

output_directory = 'lma_features/kin_5frame/'

e_meta_file_path = "EMOTIONS_META_KIN_5FRAME.txt"
f_meta_file_path = "FILES_META_KIN_5FRAME.txt"

files = []

# Read  META File
with open(f_meta_file_path, 'r') as r:
    for file in r.readlines():
        files.append(file.replace("\n",""))


input_directory_2 = 'kin_mocap_reduced/'

output_directory_2 = 'lma_features/kin_05sec/'

e_meta_file_path_2 = "EMOTIONS_META_KIN_05SEC.txt"
f_meta_file_path_2 = "FILES_META_KIN_05SEC.txt"

files_2 = []

# Read  META File
with open(f_meta_file_path_2, 'r') as r:
    for file in r.readlines():
        files_2.append(file.replace("\n",""))

  
from lma_extractor import LMAExtractor

###
ACT_STEPTIME  = 1./30.
SUBSTEPS = 20
VIS_STEP = 1.0/150
SIM_STEPTIME  = ACT_STEPTIME / SUBSTEPS
###

class LMAMocapEnv():
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

      self._visual = HumanoidKinNoVis(self._skeleton, self._model)
      self._char = self._visual.add_character("mocap")

      self._pybullet_client = self._visual._pybullet_client



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

    def step(self):
      speed = 1.0
      phase = 0
      
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
        
        links_pos[link_name] = l_info[4]
        links_orn[link_name] = l_info[5]

      pose = np.array(pose)
      vel = self._skeleton.padVel(vel)
      
      return pose, vel, links_pos, links_orn, vel_dict


def extract_features(mocap_file, output_file, model, emotion):
  env = LMAMocapEnv(mocap_file, None, model)

  env.reset()

  #lma_extractor = LMAExtractor(env,env._mocap._durations[0], append_to_file=False, outfile=output_file, label=emotion, ignore_amount=0, pool_rate=0.5)
  lma_extractor = LMAExtractor(env,env._mocap._durations[0], append_to_file=False, outfile=output_file, label=emotion, ignore_amount=0, pool_rate=-1)

  while True:
    # LMA Features
    if(not env.has_looped):
      lma_extractor.record_frame()
    else:
      # We reached the end of the mocap so we can stop recording
      lma_extractor.write_lma_features()
      break

    env.step()

def extract_features_2(mocap_file, output_file, model, emotion):
  env = LMAMocapEnv(mocap_file, None, model)

  env.reset()

  lma_extractor = LMAExtractor(env,env._mocap._durations[0], append_to_file=False, outfile=output_file, label=emotion, ignore_amount=0, pool_rate=0.5)
  #lma_extractor = LMAExtractor(env,env._mocap._durations[0], append_to_file=False, outfile=output_file, label=emotion, ignore_amount=0, pool_rate=-1)

  while True:
    # LMA Features
    if(not env.has_looped):
      lma_extractor.record_frame()
    else:
      # We reached the end of the mocap so we can stop recording
      lma_extractor.write_lma_features()
      break

    env.step()


if __name__=="__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str, default='humanoid3d', help="model")
  args = parser.parse_args()

  ## 5 FRAMES ##

  # Overwrite META File
  f_meta_file = open(f_meta_file_path, "a")
  e_meta_file = open(e_meta_file_path, "a")

  for filename in os.listdir(input_directory):
    f = os.path.join(input_directory, filename)
    if os.path.isfile(f):
        print("New File: " + f + "\n")

        emotion = filename.split("_")[0]

        o = os.path.join(output_directory, filename)

        if(os.path.exists(o) or o in files): # To avoid repeats/allow to stop and resume mass extractions
            continue

        f_meta_file.write(o + "\n")
        e_meta_file.write(emotion + "\n")

        extract_features(f, o, args.model, emotion)

        print("============================================\n")

  f_meta_file.close()
  e_meta_file.close()


  # 0.5 SEC (15 FRAMES) ##

  f_meta_file_2 = open(f_meta_file_path_2, "a")
  e_meta_file_2 = open(e_meta_file_path_2, "a")

  for filename in os.listdir(input_directory):
    f = os.path.join(input_directory, filename)
    if os.path.isfile(f):
        print("New File: " + f + "\n")

        emotion = filename.split("_")[0]

        o = os.path.join(output_directory_2, filename)

        if(os.path.exists(o) or o in files_2): # To avoid repeats/allow to stop and resume mass extractions
            continue

        f_meta_file_2.write(o + "\n")
        e_meta_file_2.write(emotion + "\n")

        extract_features_2(f, o, args.model, emotion)

        print("============================================\n")

  f_meta_file.close()
  e_meta_file.close()