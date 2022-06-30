from abc import ABC, abstractmethod
import numpy as np
import time
import argparse

from utils.humanoid_kin import HumanoidSkeleton
from utils.humanoid_mocap import HumanoidMocap
from utils.humanoid_vis import HumanoidVis
from utils.humanoid_no_vis import HumanoidNoVis

from sim import engine_builder
import threading

# timesteps
ACT_STEPTIME  = 1./30.
SUBSTEPS = 20
VIS_STEPTIME = 1.0/150
SIM_STEPTIME  = ACT_STEPTIME / SUBSTEPS

# task and contact
allowed_contacts = {
        "walk":     "walk.txt",
        "cartwheel":"cartwheel.txt",
        "crawl":    "crawl.txt",
        "roll":     "roll.txt",
        "knee":     "knee.txt",
        }

class BaseEnv(ABC):
  """
  environment abstraction

  __init__
  - build_action_bound (abstract)

  get_state_size (abstract)
  get_action_size (abstract)

  get_reset_data (abstract)

  reset (abstract)

  step
  - set_action (abstract)
  - check_valid_episode
  - record_state (abstract)
  - record_info  (abstract)
  - calc_reward  (abstract)
  - is_episode_end

  check_terminate (abstract)
  check_fall
  check_wrap_end
  check_time_end

  set_mode
  set_task_t

  """
  def __init__(self, task, seed=0, model="humanoid3d", engine="pybullet", contact="walk",
               self_collision=True, enable_draw=False,
               record_contact=False, record_torques=False):
    self._task = task
    self._rand = np.random.RandomState(seed)
    self._model= model

    self._ctrl_step = ACT_STEPTIME
    self._nsubsteps = SUBSTEPS
    self._sim_step = self._ctrl_step / self._nsubsteps

    self._enable_draw = enable_draw
    self._vis_step = VIS_STEPTIME
    self._follow_character = True

    self._record_contact = record_contact

    # initialize kinematic parts
    char_file = "data/characters/%s.txt" % self._model
    ctrl_file = "data/controllers/%s_ctrl.txt" % self._model
    motion_file = "data/motions/%s_%s.txt" % (self._model, self._task)
    self._skeleton = HumanoidSkeleton(char_file, ctrl_file)
    self._skeleton_2 = HumanoidSkeleton(char_file, ctrl_file)
    self._mocap = HumanoidMocap(self._skeleton, motion_file)
    self.curr_phase = None

    # initialize visual parts, need to be the first when using pybullet environment
    if self._enable_draw:
      # draw timer
      self._prev_clock = time.perf_counter()
      self._left_time = 0

      # visual part
      self._visual = HumanoidVis(self._skeleton, self._model)
      cnt, pose, vel = self._mocap.slerp(0)
      self._sim_char = self._visual.add_character("sim", [227/255, 170/255, 14/255, 1])

      #self._kin_char = self._visual.add_character("kin", [1, 1, 1, 0.4])
      self._kin_char = self._visual.add_character("kin", [44/255, 160/255, 44/255, 1])

      self._visual.camera_follow(self._sim_char, 2, 180, 0)

      self._no_visual = HumanoidNoVis(self._skeleton_2, self._model)
      self._pybullet_client_no_vis = self._no_visual._pybullet_client
      self._char_no_vis = self._no_visual.add_character("mocap", [44/255, 160/255, 44/255, 1])
      
      # TODO: REMOVE THIS (DEBUG)
      #self._left_wrist_height = self._visual._pybullet_client.addUserDebugParameter("lWrist y", -2, 2, 0.0)
      #self.desired_wrist_height = None
    
    ## INVERSE KINEMATICS & MOTION SYNTHESIS ##
    self._mocap_frames = []
    self._ik_frames = []

    self._gui = None
    self._ms = None
    self._synthesizing = False

    # INVERSE KINEMATICS #
    self.ik_joint_mapping = {"chest": [2,1,0],
             "neck": [5,4,3],
             "right_hip": [16,15,14],
             "right_knee": [17],
             "right_ankle": [20, 19, 18],
             "right_shoulder": [8,7,6],
             "right_elbow": [9],
             "left_hip": [23, 22, 21],
             "left_knee": [24],
             "left_ankle": [27, 26, 25],
             "left_shoulder": [12,11,10],
             "left_elbow": [13],}

    self.ik_link_mapping = {
      1 : 4, # Chest
      2 : 8, # Neck
      6 : 12, # Right Shoulder
      7 : 13, # Right Elbow
      8 : 14, # Right Wrist
      12 : 18, # Left Shoulder
      13 : 19, # Left Elbow
      14 : 20, # Left Wrist
      4 : 25, # Right Knee
      5 : 29, # Right Ankle
      10 : 34, # Left Knee
      11 : 38 # Left Ankle
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

    ###########################################

    # initialize simulation parts
    self._engine = engine_builder(engine, self._skeleton, self_collision, self._sim_step, self._model)

    contact_parser = argparse.ArgumentParser()
    contact_parser.add_argument("N", type=int, nargs="+", help="allowed contact body ids")
    contact_file = "data/contacts/%s_%s" % (self._model, allowed_contacts[contact])
    with open(contact_file) as fh:
      s = fh.read().split()
      args = contact_parser.parse_args(s)
      allowed_body_ids = args.N
    self._engine.set_allowed_fall_contact(allowed_body_ids)
    self._contact_ground = False

    # joint torque monitor # debug use
    self._monitor_joints = record_torques
    if self._monitor_joints:
      self._monitored_joints = list(range(self._skeleton.num_joints))
      self._engine.set_monitored_joints(self._monitored_joints)

    # initialize reinfrocement learning action bound
    self.a_min, self.a_max = self.build_action_bound()

    self._max_t = max(20, self._mocap._cycletime)
    self._task_t = 0.5
    self._mode = 0   # 0 for max_t and 1 for task_t

  @abstractmethod
  def get_state_size(self):
    assert(False and "not implemented")

  @abstractmethod
  def get_action_size(self):
    assert(False and "not implemented")

  @abstractmethod
  def build_action_bound(self):
    assert(False and "not implemented")

  @abstractmethod
  def get_reset_data(self):
    assert(False and "not implemented")

  @abstractmethod
  def reset(self, phase=None, state=None):
    """ Reset environment

      Inputs:
        phase
        state

      Outputs:
        obs
    """
    assert(False and "not implemented")

  def step(self, action):
    """ Step environment

      Inputs:
        action

      Outputs:
        ob
        r
        done
        info
    """
    self.set_action(action)

    self._contact_forces = []

    self._joint_torques = []

    # if not terminated during simulation
    self._contact_ground = False
    for i in range(self._nsubsteps):
      self.update()
      if self.check_fall():
        self._contact_ground = True

    is_fail = self.check_terminate()
    ob = self.record_state()
    r = 0 if is_fail else self.calc_reward()
    done = is_fail or self.is_episode_end()
    info = self.record_info()

    return ob, r, done, info

  @abstractmethod
  def set_action(self, action):
    assert(False and "not implemented")

  def update(self):
    # update draw
    if self._enable_draw and self.time_to_draw(self._sim_step):
      self.update_draw()

    self._engine.step_sim(self._sim_step)
    if self._record_contact:
      self.update_contact_forces()

    if self._monitor_joints:
      self.update_joint_torques()

    self.post_update()

  def time_to_draw(self, timestep):
    self._left_time -= timestep
    if self._left_time < 0:
      self._left_time += self._vis_step
      return True
    else:
      return False

  def apply_motion_synthesis(self):
    if(self._ms != None):
        self._synthesizing = True
        self._gui.change_motion_synthesizer_status(1)
        new_process = threading.Thread(target=self.compute_motion_synthesis_multithreaded, args=(self._gui.get_pad(), ))
        new_process.start()

  def compute_motion_synthesis_multithreaded(self, pad):
    self._ms.set_desired_pad(pad)

    self._ms.compute_coefficients()

    self._ik_frames = ["hi"]

    self._synthesizing = False 

  def update_draw(self):
    # synchronize sim pose and kin pose
    sim_pose, sim_vel = self._engine.get_pose()
    kin_t = self.curr_phase * self._mocap._cycletime
    cnt, kin_pose, kin_vel = self._mocap.slerp(kin_t)

    # offset kinematic pose
    kin_pose[:3] += cnt * self._mocap._cyc_offset
    kin_pose[0] += 1.5

    # wait until time
    time_remain = self._vis_step - (time.perf_counter() - self._prev_clock)
    if time_remain > 0:
      time.sleep(time_remain)
    self._prev_clock = time.perf_counter()

    # Inverse Kinematics #
    if(self._ik_frames != [] and not self._synthesizing):
      self._no_visual.set_pose(self._char_no_vis, sim_pose, sim_vel)
      frame = self.get_pose_and_links_no_visual()[2]
      #frame = self._engine.get_pose_and_links()[2]

      gen_pose = self._ms.convert_single_frame(frame, self.counter, sim_pose)

      indices = []
      for i in gen_pose["mocap"]:
        if(i in self.name_to_index_mapping):
          indices.append(self.ik_link_mapping[self.name_to_index_mapping[i]])
       
      sim_pose = self._engine.compute_inverse_kinematics(sim_pose, indices, gen_pose)

    # draw on window
    self._visual.set_pose(self._sim_char, sim_pose, sim_vel)
    self._visual.set_pose(self._kin_char, kin_pose, kin_vel)

    # adjust cameral pose
    if self._follow_character:
      self._visual.camera_follow(self._sim_char)
    
    self.counter += 1

  def get_pose_and_links_no_visual(self):
      return self._get_full_model_pose_no_vis(self._no_visual.characters["mocap"])

  def _get_full_model_pose_no_vis(self, phys_model):
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
      pos, orn = self._pybullet_client_no_vis.getBasePositionAndOrientation(self._no_visual.characters["mocap"])
      linvel, angvel = self._pybullet_client_no_vis.getBaseVelocity(self._no_visual.characters["mocap"])
      pose += pos
      if orn[3] < 0:
        orn = [-orn[0], -orn[1], -orn[2], -orn[3]]
      pose.append(orn[3])  # w
      pose += orn[:3] # x, y, z
      vel += linvel
      vel += angvel

      vel_dict["root"] = [linvel, angvel]

      for i in range(self._skeleton_2.num_joints):
        j_info = self._pybullet_client_no_vis.getJointStateMultiDof(phys_model, i)
        orn = j_info[0]
        if len(orn) == 4:
          pose.append(orn[3])  # w
          pose += orn[:3] # x, y, z
        else:
          pose += orn
        vel += j_info[1]

        l_info = self._pybullet_client_no_vis.getJointInfo(phys_model, i)
        
        if(not l_info[12].decode('UTF-8') == "root"):
          vel_dict[l_info[12].decode('UTF-8')] = j_info[1]
        
        link_name = l_info[12].decode('UTF-8')
        
        l_info = self._pybullet_client_no_vis.getLinkState(phys_model, i)
        
        links_pos[link_name] = l_info[4]
        links_orn[link_name] = l_info[5]

      pose = np.array(pose)
      vel = self._skeleton_2.padVel(vel)
      
      return pose, vel, links_pos, links_orn, vel_dict


  def update_contact_forces(self):
    # draw contact forces
    contact_forces = self._engine.report_contact_force()
    self._contact_forces.append(contact_forces)

  def draw_contact_force(self):
    contact_forces = self._engine.report_contact_force()
    for part, pos, force in contact_forces:
      self._visual.visual_force(pos, force)

  def update_joint_torques(self):
    jt = self._engine.get_monitored_joint_torques()
    self._joint_torques.append(jt)

  def draw_joint_torques(self):
    jt = self._engine.get_monitored_joint_torques()
    pose, vel = self._engine.get_pose()
    self._skeleton.set_pose(pose)
    for i, torque in zip(self._monitored_joints, jt):
      pos = self._skeleton.get_joint_pos(i)
      self._visual.visual_force(pos, np.array(torque))

  @abstractmethod
  def record_info(self):
    assert(False and "not implemented")

  @abstractmethod
  def post_update(self):
    assert(False and "not implemented")

  @abstractmethod
  def record_state(self):
    assert(False and "not implemented")

  @abstractmethod
  def calc_reward(self):
    assert(False and "not implemented")

  def is_episode_end(self):
    """ Check if episode is end
      episode will normally end if time excced max time
    """
    return self.check_wrap_end() or self.check_time_end()

  @abstractmethod
  def check_terminate(self):
    """ Check if simulation or task failed
    """
    assert(False and "not implemented")

  def check_fall(self):
    """ Check if any not allowed body part contact ground
    """
    # check fail
    return self._engine.check_fall()

  def check_valid_episode(self):
    # TODO check valid
    #return self._core.CheckValidEpisode()
    return True

  def check_time_end(self):
    if self._mode == 0:
      return self._engine.t >= self._task_t
    elif self._mode == 1:
      return self._engine.t >= self._max_t
    else:
      assert(False and "no supported mode")

  def check_wrap_end(self):
    """ Check if time is up for non-wrap motions
        True if motion is non-wrap and reaches phase 1.0
        False if motion is wrap or hasn't reaches phase 1.0
    """
    loop_term = not self._mocap._is_wrap and self.curr_phase > 1.0
    return loop_term

  def set_mode(self, mode):
    # 0 for train, 1 for test using max time
    assert(mode >=0 and mode < 2)
    self._mode = mode

  def set_task_t(self, t):
    """ Set the max t an episode can have under training mode
    """
    self._task_t = min(t, self._max_t)

  def set_max_t(self, t):
    """ Set the max t an episode can have under test mode
    """
    self._max_t = t

  def close(self):
    self._engine.close()
