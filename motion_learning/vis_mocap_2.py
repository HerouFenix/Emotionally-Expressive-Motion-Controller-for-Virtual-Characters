## THIS VERSION SYNTHESIZES EACH FRAME INDIVIDUALLY ##

from concurrent.futures import process
from tkinter import DISABLED, END
from tkinter.font import NORMAL
import numpy as np
import math
from utils import bullet_client
from utils.humanoid_kin import HumanoidSkeleton
from utils.humanoid_mocap import HumanoidMocap
from utils.humanoid_vis import HumanoidVis
import pybullet as p1

from multiprocessing import Process
import threading

#from IPython import embed
import time

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
  
from lma_extractor import LMAExtractor
from emotion_classifier import EmotionClassifier
from gui_manager import GUIManager
from inverse_kinematics import IKSolver
from motion_synthesizer import MotionSynthesizer

###
ACT_STEPTIME  = 1./30.
SUBSTEPS = 20
VIS_STEP = 1.0/150
SIM_STEPTIME  = ACT_STEPTIME / SUBSTEPS
###

class VisMocapEnv():
    def __init__(self, mocap_file, pybullet_client=None, model="humanoid3d", motion_synthesizer = None):
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
      self._ms = motion_synthesizer
      self._gui = None

      self._synthesizing = False
      self._synthesized = False

      # INVERSE KINEMATICS #
      self._ik_solver = IKSolver()
      self.ik_joint_mapping = {"chest": [0,1,2],
               "neck": [3,4,5],

               "right_hip": [14,15,16],
               "right_knee": [17],
               "right_ankle": [18, 19, 20],
               "right_shoulder": [6,7,8],
               "right_elbow": [9],

               "left_hip": [21, 22, 23],
               "left_knee": [24],
               "left_ankle": [25, 26, 27],
               "left_shoulder": [10,11,12],
               "left_elbow": [13],}
      self.ik_link_mapping = {
        1 : 4, # Chest
        2 : 7, # Neck
        6 : 10, # Right Shoulder
        7 : 11, # Right Elbow
        8 : 12, # Right Wrist

        12 : 15, # Left Shoulder
        13 : 16, # Left Elbow
        14 : 17, # Left Wrist

        4 : 21, # Right Knee
        5 : 24, # Right Ankle

        10 : 28, # Left Knee
        11 : 31 # Left Ankle
      }
      self.name_to_index_mapping = {
        "chest":1,
        "neck": 2,
        "right_shoulder": 6,
        "right_elbow": 7,
        "right_wrist": 8,
        "left_shoulder": 12,
        "left_elbow": 13,
        "left_wrist": 14,
        "right_knee": 4,
        "right_ankle": 5,
        "left_knee": 10,
        "left_ankle": 11
      }

      self.counter = 0


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


    def compute_inverse_kinematics_single(self,frame, links, desired_pos):
      self._ik_solver.updatePose(frame)

      #self._ik_solver.adjustBase(desired_pos["mocap"]["root"][1])   

      pos = [desired_pos["mocap"]["neck"], desired_pos["mocap"]["left_wrist"], desired_pos["mocap"]["right_wrist"], desired_pos["mocap"]["left_elbow"], desired_pos["mocap"]["right_elbow"]]
      #jointPoses = self._ik_solver.calculateKinematicSolution2(links, pos)

      # Neck
      jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]
      jointPoses = self._ik_solver.calculateKinematicSolution(links[0], pos[0], desiredOrientation=None, jd=jd)

      # Wrists
      jd = [10000, 10000, 10000, 10000, 10000, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]
      #jointPoses = self._ik_solver.calculateKinematicSolution2([links[1],links[2]], [pos[1], pos[2]], jd=jd)

      # Elbows
      jd = [10000, 10000, 10000, 10000, 10000, 10000,10000, 10000, 10000, 10000, 10000, 0.1, 0.1, 10000, 10000, 10000, 0.1, 0.1, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]
      #jointPoses = self._ik_solver.calculateKinematicSolution2([links[3],links[4]], [pos[3], pos[4]], jd=jd)


      ik_frame = []
      
      # Add Base info from frame
      #base_info = [frame[0], desired_pos[gen_index]["mocap"]["root"][1], frame[2], frame[3], frame[4], frame[5], frame[6]]
      base_info = [frame[0], frame[1], frame[2], frame[3], frame[4], frame[5], frame[6]]
      ik_frame += list(base_info)
        
      # Add the rest of our frame info from our computed IK solution
      chest_rotation = self._pybullet_client.getQuaternionFromEuler([jointPoses[self.ik_joint_mapping["chest"][0]], jointPoses[self.ik_joint_mapping["chest"][1]], jointPoses[self.ik_joint_mapping["chest"][2]]])
      neck_rotation = self._pybullet_client.getQuaternionFromEuler([jointPoses[self.ik_joint_mapping["neck"][0]], jointPoses[self.ik_joint_mapping["neck"][1]], jointPoses[self.ik_joint_mapping["neck"][2]]])
      right_hip_rotation = self._pybullet_client.getQuaternionFromEuler([jointPoses[self.ik_joint_mapping["right_hip"][0]], jointPoses[self.ik_joint_mapping["right_hip"][1]], jointPoses[self.ik_joint_mapping["right_hip"][2]]])
      right_knee_rotation = [jointPoses[self.ik_joint_mapping["right_knee"][0]]]
      right_ankle_rotation = self._pybullet_client.getQuaternionFromEuler([jointPoses[self.ik_joint_mapping["right_ankle"][0]], jointPoses[self.ik_joint_mapping["right_ankle"][1]], jointPoses[self.ik_joint_mapping["right_ankle"][2]]])
      right_shoulder_rotation = self._pybullet_client.getQuaternionFromEuler([jointPoses[self.ik_joint_mapping["right_shoulder"][0]], jointPoses[self.ik_joint_mapping["right_shoulder"][1]], jointPoses[self.ik_joint_mapping["right_shoulder"][2]]])
      right_elbow_rotation = [jointPoses[self.ik_joint_mapping["right_elbow"][0]]]
      left_hip_rotation = self._pybullet_client.getQuaternionFromEuler([jointPoses[self.ik_joint_mapping["left_hip"][0]], jointPoses[self.ik_joint_mapping["left_hip"][1]], jointPoses[self.ik_joint_mapping["left_hip"][2]]])
      left_knee_rotation = [jointPoses[self.ik_joint_mapping["left_knee"][0]]]
      left_ankle_rotation = self._pybullet_client.getQuaternionFromEuler([jointPoses[self.ik_joint_mapping["left_ankle"][0]], jointPoses[self.ik_joint_mapping["left_ankle"][1]], jointPoses[self.ik_joint_mapping["left_ankle"][2]]])
      left_shoulder_rotation = self._pybullet_client.getQuaternionFromEuler([jointPoses[self.ik_joint_mapping["left_shoulder"][0]], jointPoses[self.ik_joint_mapping["left_shoulder"][1]], jointPoses[self.ik_joint_mapping["left_shoulder"][2]]])
      left_elbow_rotation = [jointPoses[self.ik_joint_mapping["left_elbow"][0]]]

      ik_frame += [chest_rotation[3], chest_rotation[0], chest_rotation[1], chest_rotation[2]]
      ik_frame += [neck_rotation[3], neck_rotation[0], neck_rotation[1], neck_rotation[2]]
      ik_frame += [right_hip_rotation[3], right_hip_rotation[0], right_hip_rotation[1], right_hip_rotation[2]]
      ik_frame += right_knee_rotation
      ik_frame += [right_ankle_rotation[3], right_ankle_rotation[0], right_ankle_rotation[1], right_ankle_rotation[2]]
      ik_frame += [right_shoulder_rotation[3], right_shoulder_rotation[0], right_shoulder_rotation[1], right_shoulder_rotation[2]]
      ik_frame += right_elbow_rotation
      ik_frame += [left_hip_rotation[3], left_hip_rotation[0], left_hip_rotation[1], left_hip_rotation[2]]
      ik_frame += left_knee_rotation
      ik_frame += [left_ankle_rotation[3], left_ankle_rotation[0], left_ankle_rotation[1], left_ankle_rotation[2]]
      ik_frame += [left_shoulder_rotation[3], left_shoulder_rotation[0], left_shoulder_rotation[1], left_shoulder_rotation[2]]
      ik_frame += left_elbow_rotation

      #ik_frame = np.asarray(ik_frame)
      ik_frame = frame
      return ik_frame

    def compute_and_apply_motion_synthesis(self):
      if(self._ms != None):
        self._synthesizing = True
        self._gui.change_motion_synthesizer_status(1)
        new_process = threading.Thread(target=self.compute_and_apply_motion_synthesis_multithreaded, args=(self._gui.get_pad(), ))
        new_process.start()

    def compute_and_apply_motion_synthesis_multithreaded(self, pad):
      #self._ms.set_desired_pad(pad)
      self._ms.set_reference_lma()

      self._ms.compute_coefficients()
      #motion_changes = self._ms.get_motion_changes()

      #indices = []
      #for i in motion_changes[0]["mocap"]:
      #  if(i in self.name_to_index_mapping):
      #    indices.append(self.ik_link_mapping[self.name_to_index_mapping[i]])
      
      #ik_frames = self.compute_inverse_kinematics(indices, motion_changes)
      #self._mocap._ik_frames = ik_frames

      self._synthesizing = False
      self._synthesized = True

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
      self._ik_solver.updatePose(pose)

      if(self._synthesized):
        frame = self.get_pose_and_links()[2]

        gen_pose = self._ms.convert_single_frame(frame, self.counter)
        indices = []
        for i in gen_pose["mocap"]:
          if(i in self.name_to_index_mapping):
            indices.append(self.ik_link_mapping[self.name_to_index_mapping[i]])
        
        pose = self.compute_inverse_kinematics_single(pose, indices, gen_pose)

        self._visual.set_pose(self._char, pose, vel)

      self.counter += 1

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

    def get_pose_and_links_world(self):
      return self._get_full_model_pose_world(self._visual.characters["mocap"])
    
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

    def _get_full_model_pose_world(self, phys_model):
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


def show_mocap(mocap_file, model, record_lma='', predict_emotion=True, record_mocap=''):
  env = VisMocapEnv(mocap_file, None, model)
  #env._mocap.show_com()
  env.reset()
  if(record_lma != ""):
    if(record_mocap != ''):
      lma_extractor = LMAExtractor(env, env._mocap._durations[0], record_lma, append_to_file=True, pool_rate=-1, write_mocap=True, write_mocap_file=record_mocap)
    else:
      lma_extractor = LMAExtractor(env, env._mocap._durations[0], record_lma, append_to_file=True, pool_rate=-1)
  else:
    lma_extractor = LMAExtractor(env, env._mocap._durations[0], append_to_file=False, label="NONE", pool_rate=-1)

  if(predict_emotion):
    gui = GUIManager()
    gui.change_animation_status(2)
    gui.change_emotion_prediction_status(0)

    gui.start_motion_synthesis.configure(command=env.compute_and_apply_motion_synthesis)
    gui.start_motion_synthesis.configure(state=DISABLED)

    #gui.update()
    current_emotion = [0.0, 0.0, 0.0]
    emotion_predictor = EmotionClassifier()


    ms = MotionSynthesizer()
    env._ms = ms

    env._gui = gui
  
  processes = []
  has_looped_once = False

  gui.update()

  while True:
    # LMA Features
    if(env._synthesizing):
      gui.change_motion_synthesizer_status(1)
      gui.start_motion_synthesis.configure(state=DISABLED)
    else:
      gui.change_motion_synthesizer_status(0)
      if(has_looped_once):
        gui.start_motion_synthesis.configure(state=NORMAL)

    if(not env.has_looped):
      lma_extractor.record_frame()

      if(not has_looped_once and predict_emotion):
        gui.change_animation_status(0)

      # Every 10 LMA features, run predictor
      if(predict_emotion):
        if(len(lma_extractor.get_lma_features()) >= 10):
          new_process = threading.Thread(target=emotion_predictor.predict_emotion_coordinates, args=(lma_extractor.get_lma_features(),current_emotion,))
          #new_process = Process(target=emotion_predictor.predict_emotion_coordinates, args=(lma_extractor.get_lma_features(),))
          processes.append(new_process)
          new_process.start()

          lma_extractor.clear()
      
        # Update GUI
        gui.change_emotion_coordinates(current_emotion[0], current_emotion[1], current_emotion[2])
        gui.change_emotion_prediction_status(1)
        #gui.update()

    else:
      if(predict_emotion):
        
        # Check if there are still lma features that didn't get emotion classified. If so, predict them as a last batch
        if(len(lma_extractor.get_lma_features()) > 0):
          new_process = threading.Thread(target=emotion_predictor.predict_emotion_coordinates, args=(lma_extractor.get_lma_features(), ))
          processes.append(new_process)
          new_process.start()

          lma_extractor.clear()

        # Wait for all child processes to be done before computing the final coordinates
        for p in processes:
          p.join()

        lma_extractor.clear()
        processes = []

        predicted_p, predicted_a, predicted_d = emotion_predictor.predict_final_emotion()
        current_emotion = [predicted_p, predicted_a, predicted_d]
        emotion_predictor.clear()

        gui.change_emotion_coordinates(current_emotion[0], current_emotion[1], current_emotion[2])
        gui.change_emotion_prediction_status(2)

        if(ms._mocap == []):
          ms.set_current_lma(lma_extractor.lma_full)
          ms.set_current_mocap(lma_extractor.mocap_full)

          gui.start_motion_synthesis.configure(state=NORMAL)

        lma_extractor.clear_full()

      if(env._mocap._is_wrap):
        env.reset()
        env.has_looped = False

        if(predict_emotion):
          gui.change_animation_status(1)
          #gui.update()

        has_looped_once = True

        if(lma_extractor._append_to_file):
          lma_extractor._append_to_file = False
        if(lma_extractor._write_mocap):
          lma_extractor._write_mocap = False

      else:
        if(predict_emotion):
          gui.change_animation_status(2)
          #gui.update()

        while(True):
          continue #NOTE: this is just here so that the window doesn't immediately close (so that the user has to manually close the window to terminate the program)
      
      env.counter = 0

    env.step()
    gui.update()

if __name__=="__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--mocap", type=str, default='data/motions/humanoid3d_jump.txt', help="task to perform")
  parser.add_argument("--model", type=str, default='humanoid3d', help="model")
  parser.add_argument("--record_lma", type=str, default='', help="specify a file name if you want to store the lma features on a file")
  parser.add_argument("--record_mocap", type=str, default='', help="specify a file name if you want to store the mocap on a file")
  
  parser.add_argument("--predict_emotion",default=True, action="store_true" , help="specify whether you want to output the predicted emotional coordinates")

  args = parser.parse_args()

  show_mocap(args.mocap, args.model, args.record_lma, args.predict_emotion, args.record_mocap)
