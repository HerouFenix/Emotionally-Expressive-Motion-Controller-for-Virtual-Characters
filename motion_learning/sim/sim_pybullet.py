from .sim_engine import SimEngine
from .sim_engine import ACT_STEPTIME, SUBSTEPS, VIS_STEPTIME
import pybullet as p1
from utils import bullet_client
from utils.humanoid_kin import JointType
import time
import math
import numpy as np

from inverse_kinematics import IKSolver

class PyBulletEngine(SimEngine):
  """ simulation engine for character controll

      using pybullet as backend
  """

  def __init__(self, skeleton, self_collision, sim_timestep, model="humanoid3d"):
    self._skeleton = skeleton
    self._self_collision = self_collision
    self._sim_step = sim_timestep

    self.t = None
    self._pybullet_client  =  bullet_client.BulletClient(p1.DIRECT)
    self._model_name = model # should be humanoid3d or atlas

    # set simulation environment parameters
    self._pybullet_client.setGravity(0,-9.8,0)
    self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=10)
    self._pybullet_client.setTimeStep(self._sim_step)
    self._pybullet_client.setPhysicsEngineParameter(numSubSteps=1)

    # load ground
    self._init_ground()

    # load sim_char
    self._init_sim(self_collision)
    self._init_spd()

    # contact pairs
    self._pts = None

    # joint torques
    self._monitored_joints = None


    # INVERSE KINEMATICS #
    self._ik_solver = IKSolver()
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
               

  def _init_ground(self):
    # load ground plane
    z2y = self._pybullet_client.getQuaternionFromEuler([-math.pi*0.5,0,0])
    self._plane = self._pybullet_client.loadURDF("./data/urdf/plane_implicit.urdf",[0,0,0],z2y, useMaximalCoordinates=True)
    self._pybullet_client.changeDynamics(self._plane, linkIndex=-1, lateralFriction=0.9)

  def _init_sim(self, self_collision):
    """ Initialize character for simulation

      _sim_model
    """
    # load simulation model
    flags = self._pybullet_client.URDF_MAINTAIN_LINK_ORDER
    if self_collision:
      flags = flags | self._pybullet_client.URDF_USE_SELF_COLLISION

    if self._model_name == "humanoid3d":
      self._sim_model = self._pybullet_client.loadURDF(
        "./data/urdf/humanoid/humanoid.urdf", [0,0.85,0],
        useFixedBase=False, flags=flags)
    elif self._model_name == "atlas":
      self._sim_model = self._pybullet_client.loadURDF(
        "./data/urdf/atlas/atlas.urdf", [0,0,0.929],
        useFixedBase=False, flags=flags)
    else:
      raise NotImplementedError

    self._pybullet_client.changeDynamics(self._sim_model, -1, lateralFriction=0.9)
    for j in range (self._pybullet_client.getNumJoints(self._sim_model)):
      self._pybullet_client.changeDynamics(self._sim_model, j, lateralFriction=0.9)

    self._pybullet_client.changeDynamics(self._sim_model, -1, linearDamping=0, angularDamping=0)

    # initialize joints' controllers
    jointFrictionForce = 0
    for j, jtype in enumerate(self._skeleton.joint_types):
      if jtype is JointType.BASE:
        pass
      elif jtype is JointType.FIXED:
        pass
      elif jtype is JointType.REVOLUTE:
        self._pybullet_client.setJointMotorControl2(self._sim_model, j, self._pybullet_client.POSITION_CONTROL, targetPosition=0, positionGain=0, targetVelocity=0,force=jointFrictionForce)
      elif jtype is JointType.SPHERE:
        self._pybullet_client.setJointMotorControlMultiDof(self._sim_model, j, self._pybullet_client.POSITION_CONTROL,targetPosition=[0,0,0,1], targetVelocity=[0,0,0], positionGain=0,velocityGain=1,force=[jointFrictionForce,jointFrictionForce,jointFrictionForce])

  def _init_spd(self):
    """ Initialize spd settings

    """
    # spd control set up
    spd_joint_ids = []
    spd_joint_dofs = []
    spd_joint_kps = []
    spd_joint_kds = []
    spd_joint_force_limits = []

    index = 7 # start point to read self._skeleton.kp / kd / torque_lim
    for i in range(1, len(self._skeleton.joint_dof)):
      nDof = self._skeleton.joint_dof[i]
      if nDof == 0:
        continue
      spd_joint_ids.append(i)
      spd_joint_dofs.append(nDof)
      spd_joint_kps.append(self._skeleton.kp[index])
      spd_joint_kds.append(self._skeleton.kd[index])
      if nDof == 4:
        spd_joint_force_limits.append([
            self._skeleton.torque_lim[index + 0],
            self._skeleton.torque_lim[index + 1],
            self._skeleton.torque_lim[index + 2]
        ])
      elif nDof == 1:
        spd_joint_force_limits.append([self._skeleton.torque_lim[index + 0]])

      index += nDof

    self._spd_params = {
        "ids": spd_joint_ids,
        "dofs": spd_joint_dofs,
        "kps": spd_joint_kps,
        "kds": spd_joint_kds,
        "force_limits": spd_joint_force_limits
    }

  def set_allowed_fall_contact(self, body_ids):
    """
        only those body parts/links are allowed to touch the ground,
        otherwise the episode terminates
    """
    self._allowed_body_parts = body_ids

  def set_monitored_joints(self, joint_ids):
    self._monitored_joints = joint_ids
    for i in joint_ids:
      self._pybullet_client.enableJointForceTorqueSensor(self._sim_model, i)

  def get_monitored_joint_torques(self):
    if self._monitored_joints is None:
      return []
    js = self._pybullet_client.getJointStatesMultiDof(self._sim_model, self._monitored_joints)
    jt = list(map(lambda x: x[2][-3:], js))
    return jt

  def disable_monitor_joints(self):
    for i in range(self._pybullet_client.getNumJoints(self._sim_model)):
      self._pybullet_client.enableJointForceTorqueSensor(self._sim_model, i, 0)
    self._monitored_joints = None

  def set_kp(self, kp):
    pass

  def set_kd(self, kd):
    pass

  def set_force_limit(self, force_lim):
    pass

  def set_steptime(self, steptime, substeps):
    self._act_step = act_steptime
    self._substeps = substeps
    self._sim_step = act_steptime / substeps

  def get_pose(self):
    return self._get_model_pose(self._sim_model)

  def get_pose_and_links(self):
    return self._get_full_model_pose(self._sim_model)

  def _get_model_pose(self, phys_model):
    """ Get current pose and velocity expressed in general coordinate
      Inputs:
        phys_model

      Outputs:
        pose
        vel
    """
    pose = []
    vel = []

    # root position/orientation and vel/angvel
    pos, orn = self._pybullet_client.getBasePositionAndOrientation(self._sim_model)
    linvel, angvel = self._pybullet_client.getBaseVelocity(self._sim_model)
    pose += pos
    if orn[3] < 0:
      orn = [-orn[0], -orn[1], -orn[2], -orn[3]]
    pose.append(orn[3])  # w
    pose += orn[:3] # x, y, z
    vel += linvel
    vel += angvel

    for i in range(self._skeleton.num_joints):
      j_info = self._pybullet_client.getJointStateMultiDof(phys_model, i)
      orn = j_info[0]
      if len(orn) == 4:
        pose.append(orn[3])  # w
        pose += orn[:3] # x, y, z
      else:
        pose += orn

      vel += j_info[1]

    pose = np.array(pose)
    vel = self._skeleton.padVel(vel)
    return pose, vel

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
    #links_vel = {}
    links_orn = {}
    #links_ang = {}

    vel_dict = {}

    # root position/orientation and vel/angvel
    pos, orn = self._pybullet_client.getBasePositionAndOrientation(self._sim_model)
    linvel, angvel = self._pybullet_client.getBaseVelocity(self._sim_model)
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
      
      #if(len(l_info) > 6):
      #  links_vel[link_name] = l_info[6]
      #  links_ang[link_name] = l_info[7]
      #else:
      #  links_vel[link_name] = []
      #  links_ang[link_name] = []

    pose = np.array(pose)
    vel = self._skeleton.padVel(vel)
    
    # print(links_pos)
    return pose, vel, links_pos, links_orn, vel_dict

  def _get_base_pos(self):
    pos, orn = self._pybullet_client.getBasePositionAndOrientation(self._sim_model)
    return pos

  def set_pose(self, pose, vel):
    self._set_model_pose(pose, vel, self._sim_model)
    self.t = 0

  def _set_model_pose(self, pose, vel, phys_model, initBase=True):
    """ Set character state in physics engine
      Inputs:
        pose   np.array of float, self._skeleton.pos_dim, position of base and
               orintation of joints, represented in local frame
        vel    np.array of float, self._skeleton.vel_dim, velocity of base and
               angular velocity of joints, represented in local frame

        phys_model  pybullet model unique Id, self._sim_model or self._kin_model

        initBase bool, if set base position/orintation/velocity/angular velocity
                 as well
    """
    s = self._skeleton
    if initBase:
      pos = pose[:3]
      orn_wxyz = pose[3:7]
      orn = [orn_wxyz[1], orn_wxyz[2], orn_wxyz[3], orn_wxyz[0]]
      v   = vel[:3]
      omg = vel[3:6]
      self._pybullet_client.resetBasePositionAndOrientation(phys_model, pos, orn)
      self._pybullet_client.resetBaseVelocity(phys_model, v, omg)

    for i in range(s.num_joints):
      jtype = s.joint_types[i]
      p_off = s.pos_start[i]
      if jtype is JointType.BASE:
        pass
      elif jtype is JointType.FIXED:
        pass
      elif jtype is JointType.REVOLUTE:
        orn = [pose[p_off]]
        omg = [vel[p_off]]
        self._pybullet_client.resetJointStateMultiDof(phys_model, i, orn, omg)
      elif jtype is JointType.SPHERE:
        orn_wxyz = pose[p_off : p_off+4]
        orn = [orn_wxyz[1], orn_wxyz[2], orn_wxyz[3], orn_wxyz[0]]
        omg = vel[p_off : p_off+3]
        self._pybullet_client.resetJointStateMultiDof(phys_model, i, orn, omg)

  def set_spd_target(self, pose):
    """ Apply target pose on SPD controller
      Inputs:
        target
    """
    if pose is None:
      assert(False and "set target pose as None")

    pose = np.array(pose)
    pybullet_target = self._skeleton.pose_wxyz_to_xyzw(pose)

    self._spd_target_positions = []
    index = 7
    for dof in self._spd_params["dofs"]:
      if dof == 4:
        self._spd_target_positions.append([
            pybullet_target[index + 0],
            pybullet_target[index + 1],
            pybullet_target[index + 2],
            pybullet_target[index + 3],
        ])
      else :
        self._spd_target_positions.append([pybullet_target[index + 0]])

      index += dof

  def step_sim(self, timestep):
    self._pybullet_client.setTimeStep(timestep)
    self._apply_spd_torque()
    self._pybullet_client.stepSimulation()
    self.t += timestep
    self._pts = self._pybullet_client.getContactPoints()

  def _apply_spd_torque(self):
    """ Apply spd target pose to step simulation
    """
    control_mode = None
    if self._model_name == "humanoid3d":
      control_mode = self._pybullet_client.STABLE_PD_CONTROL
    elif self._model_name == "atlas":
      #control_mode = self._pybullet_client.POSITION_CONTROL
      control_mode = self._pybullet_client.PD_CONTROL
    elif self._model_name == "atlas_jason":
      #control_mode = self._pybullet_client.STABLE_PD_CONTROL
      control_mode = self._pybullet_client.PD_CONTROL
    else:
      raise NotImplementedError

    self._pybullet_client.setJointMotorControlMultiDofArray(
      self._sim_model,
      self._spd_params["ids"],
      control_mode,
      targetPositions= self._spd_target_positions,
      positionGains=self._spd_params["kps"],
      velocityGains=self._spd_params["kds"],
      forces=self._spd_params["force_limits"]
    )

  def check_fall(self):
    fall=False
    for p in self._pts:
      part = -1
      #ignore self-collision
      if (p[1]==p[2]):
        continue
      if (p[1]==self._sim_model):
        part=p[3]
      if (p[2]==self._sim_model):
        part=p[4]
      if (part >=0 and part not in self._allowed_body_parts):
        #print("terminating part:", part)
        fall=True

    return fall

  def check_self_contact(self):
    fall=False
    for p in self._pts:
      #self-collision
      if (p[1]==p[2]):
        return True
    return False

  def report_contact_force(self):
    """ Return contact infor during simulation

      Outputs:
        list of tuples (part, pos, force)
    """
    if self._pts is None:
      return []
    contact_forces = []
    for p in self._pts:
      part = -1
      #ignore self-collision
      if (p[1]==p[2]):
        continue
      if (p[1]==self._sim_model):
        part = p[3]
        pos = p[5]
      if (p[2]==self._sim_model):
        part = p[4]
        pos = p[6]

      fn = np.array(p[7]) * p[9]
      f0 = np.array(p[11]) * p [10]
      f1 = np.array(p[13]) * p [12]

      f = fn + f0 + f1

      # tuple (part, pos, force)
      contact_forces.append((part, pos, f))

    return contact_forces

  def close(self):
    pass


  def compute_inverse_kinematics(self,frame, links, desired_pos):
    self._ik_solver.updatePose(frame)

    self._ik_solver.adjustBase(desired_pos["mocap"]["root"][1])   

    pos = [desired_pos["mocap"]["neck"], desired_pos["mocap"]["left_wrist"], desired_pos["mocap"]["right_wrist"], desired_pos["mocap"]["left_elbow"], desired_pos["mocap"]["right_elbow"], desired_pos["mocap"]["left_ankle"], desired_pos["mocap"]["right_ankle"]]
    jointPoses = self._ik_solver.calculateKinematicSolution2(links, pos)

    ik_frame = []
      
    # Add Base info from frame
    base_info = [frame[0], desired_pos["mocap"]["root"][1], frame[2], frame[3], frame[4], frame[5], frame[6]]
    #base_info = [frame[0], frame[1], frame[2], frame[3], frame[4], frame[5], frame[6]]
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

    ik_frame = np.asarray(ik_frame)
    return ik_frame


  def compute_inverse_kinematics2(self, base_frames, extraction_framerate, links, desired_pos):
      ik_frames = []
      i = 0
      #increment = round(len(reference_frames) / len(base_frames))
      increment = 1
      first = True
      last_frame = None

      for frame in base_frames:
        if(first):
          # Ignore the first frame
          first = False
          ik_frames.append(frame)

          i += increment
          last_frame = frame
          continue
        
        if(i % extraction_framerate == 0):
          self._ik_solver.updatePose(frame)


          gen_index = next(index for index in range(
                    len(desired_pos)) if desired_pos[index]["index"] == i)

          #self._ik_solver.adjustBase(desired_pos[gen_index]["mocap"]["root"][1])          

          pos = [desired_pos[gen_index]["mocap"]["neck"], desired_pos[gen_index]["mocap"]["left_wrist"], desired_pos[gen_index]["mocap"]["right_wrist"]]
          jointPoses = self._ik_solver.calculateKinematicSolution2(links, pos)

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

          ik_frames.append(np.asarray(ik_frame))
          last_frame = ik_frame
        else:
          ik_frames.append(np.asarray(last_frame))

        
        i += increment

      return ik_frames
