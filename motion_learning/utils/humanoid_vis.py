import math
import numpy as np
from utils import bullet_client
from utils.humanoid_kin import JointType
import pybullet as p1

URDF_DIR = "data/urdf"

class HumanoidVis(object):

  def __init__(self, skeleton, model="humanoid3d"):
    self._skeleton = skeleton
    self.characters = dict()
    # init pybullet client
    self._init_physics()
    self._model = model # humanoid3d or atlas
    self.first = True
    self.c = []

  def _init_physics(self):
    self._pybullet_client =  bullet_client.BulletClient(connection_mode=p1.GUI)
    self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_GUI,0)
    self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_Y_AXIS_UP,1)

    # load ground plane
    self._pybullet_client.setAdditionalSearchPath(URDF_DIR)
    z2y = self._pybullet_client.getQuaternionFromEuler([-math.pi*0.5,0,0])
    #z2y = self._pybullet_client.getQuaternionFromEuler([0,0,0])
    self._planeId = self._pybullet_client.loadURDF("plane_implicit.urdf",[0,0,0],z2y, useMaximalCoordinates=True)

    # set simulation environment parameters
    self._pybullet_client.setGravity(0,-9.8,0)

    self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=10)
    self._pybullet_client.changeDynamics(self._planeId, linkIndex=-1, lateralFriction=0.9)

    self._pybullet_client.setTimeStep(1.0/60.0)
    self._pybullet_client.setPhysicsEngineParameter(numSubSteps=1)

  def add_character(self, name, color):
    assert(name not in self.characters.keys())
    char_id = self._new_character(color)
    self.characters[name] = char_id
    return name

  def _new_character(self, color):

    if self._model == "humanoid3d":
      kin_model = self._pybullet_client.loadURDF(
        "humanoid/humanoid.urdf", [0,0.889540259,0],
        useFixedBase=True, flags=self._pybullet_client.URDF_MAINTAIN_LINK_ORDER)
    elif self._model == "atlas":
      kin_model = self._pybullet_client.loadURDF(
        "atlas/atlas.urdf", [0,0,0],
        useFixedBase=True, flags=self._pybullet_client.URDF_MAINTAIN_LINK_ORDER)
    else:
      raise NotImplementedError

    self._pybullet_client.changeDynamics(kin_model, -1, linearDamping=0, angularDamping=0)

    # set kinematic character dymanic, collision and vision property
    kin_act_state = (self._pybullet_client.ACTIVATION_STATE_SLEEP
                    + self._pybullet_client.ACTIVATION_STATE_ENABLE_SLEEPING
                    + self._pybullet_client.ACTIVATION_STATE_DISABLE_WAKEUP)
    self._pybullet_client.setCollisionFilterGroupMask(kin_model,-1,collisionFilterGroup=0,collisionFilterMask=0)
    self._pybullet_client.changeDynamics(kin_model,-1,activationState=kin_act_state)
    self._pybullet_client.changeVisualShape(kin_model,-1, rgbaColor=color, textureUniqueId=-1)
    for j in range (self._pybullet_client.getNumJoints(kin_model)):
      self._pybullet_client.setCollisionFilterGroupMask(kin_model,j,collisionFilterGroup=0,collisionFilterMask=0)
      self._pybullet_client.changeDynamics(kin_model,j,activationState=kin_act_state)
      self._pybullet_client.changeVisualShape(kin_model,j, rgbaColor=color, textureUniqueId=-1)

    return kin_model

  def set_pose(self, char_name, pose, vel):
    """ Set character state in physics engine
      Inputs:
        pose   np.array of float, self._skeleton.pos_dim, position of base and
               orintation of joints, represented in local frame
        vel    np.array of float, self._skeleton.vel_dim, velocity of base and
               angular velocity of joints, represented in local frame

        phys_model  pybullet model unique Id, self._sim_model or self._kin_model
    """

    #{'base': -1, 'root': 0, 'chest': 1, 'neck': 2, 'right_hip': 3, 'right_knee': 4, 'right_ankle': 5, 'right_shoulder': 6, 'right_elbow': 7, 'right_wrist': 8, 'left_hip': 9, 'left_knee': 10, 'left_ankle': 11, 'left_shoulder': 12, 'left_elbow': 13, 'left_wrist': 14}

    assert(char_name in self.characters.keys())
    phys_model = self.characters[char_name]
    s = self._skeleton
    pos = pose[:3]
    orn_wxyz = pose[3:7]
    orn = [orn_wxyz[1], orn_wxyz[2], orn_wxyz[3], orn_wxyz[0]]
    v   = vel[:3]
    omg = vel[3:6]
    self._pybullet_client.resetBasePositionAndOrientation(phys_model, pos, orn)
    self._pybullet_client.resetBaseVelocity(phys_model, v, omg)

#
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
        print(self._pybullet_client.getJointInfo(phys_model, i)[1])
        print(self._pybullet_client.getJointStateMultiDof(phys_model, i)[0])
        print()
        
      elif jtype is JointType.SPHERE:
        orn_wxyz = pose[p_off : p_off+4]
        orn = [orn_wxyz[1], orn_wxyz[2], orn_wxyz[3], orn_wxyz[0]]
        omg = vel[p_off : p_off+3]
        self._pybullet_client.resetJointStateMultiDof(phys_model, i, orn, omg)

        print(self._pybullet_client.getJointInfo(phys_model, i)[1])
        print(self._pybullet_client.getJointStateMultiDof(phys_model, i)[0])
        print()

    print(pose)

    """
    assert(char_name in self.characters.keys())
    phys_model = self.characters[char_name]
    s = self._skeleton

    #TODO: Pass current pose to IKSolver ; Pass desired position to IKSolver
    #TODO: Get jointPoses from IKSolver
    #jointPoses = IKSolver.computeInverseKinematics(current_pose, desired_positions)
    jointPoses = [-1.4606487387620555, -0.861686020530726, 0.09230053991733346, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9884745715857743, 1.8293280170440733, 0.7547371658637096, -0.06251594145376464, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    mapping = {"chest": [0,1,2],
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

    for i in range(s.num_joints):
      jtype = s.joint_types[i]

      if jtype is JointType.BASE: #Root
        pass
      elif jtype is JointType.FIXED: #R/L Wrist
        pass
      elif jtype is JointType.REVOLUTE: #R/L Knee ; R/L Elbow
        joint_name = self._pybullet_client.getJointInfo(phys_model,i)[1].decode("utf-8")
        orn = [jointPoses[mapping[joint_name][0]]]
        self._pybullet_client.resetJointStateMultiDof(phys_model, i, orn)
      elif jtype is JointType.SPHERE: #Chest ; Neck ; R/L Hip ; R/L Ankle ; R/L Shoulder
        joint_name = self._pybullet_client.getJointInfo(phys_model,i)[1].decode("utf-8")
        orn = [jointPoses[mapping[joint_name][0]], jointPoses[mapping[joint_name][2]], jointPoses[mapping[joint_name][1]]] #TODO: Change this
        orn = self._pybullet_client.getQuaternionFromEuler(orn)
        self._pybullet_client.resetJointStateMultiDof(phys_model, i, orn)
    """

    #chest chest chest neck neck neck right_hip right_hip right_hip right_knee right_ankle right_ankle right_ankle right_shoulder right_shoulder right_shoulder right_elbow left_hip left_hip left_hip left_knee left_ankle left_ankle left_ankle left_shoulder left_shoulder left_shoulder left_elbow


  def camera_follow(self, char_name, dis=None, yaw=None, pitch=None, pos=None):
    assert(char_name in self.characters.keys())
    phys_model = self.characters[char_name]

    _pos = np.array(self._get_base_pose(phys_model))
    _pos[1] = 1
    _pos[0] += 0.75

    cam_info = self._pybullet_client.getDebugVisualizerCamera()
    _yaw, _pitch, _dis = cam_info[8], cam_info[9], cam_info[10]
    dis = _dis if dis is None else dis
    yaw = _yaw if yaw is None else yaw
    pitch = _pitch if pitch is None else pitch
    pos = _pos if pos is None else pos
    self._pybullet_client.resetDebugVisualizerCamera(dis, yaw, pitch, pos)

  def _get_base_pose(self, phys_model):
    pos, orn = self._pybullet_client.getBasePositionAndOrientation(phys_model)
    return pos

  def _get_joint_pose(self, phys_model, jid):
    info = self._pybullet_client.getLinkState(phys_model, jid)
    pos = info[4]
    orn = info[5]
    return pos, orn


class HumanoidKinNoVis(object):

  def __init__(self, skeleton, model="humanoid3d"):
    self._skeleton = skeleton
    self.characters = dict()
    # init pybullet client
    self._init_physics()
    self._model = model # humanoid3d or atlas

  def _init_physics(self):
    self._pybullet_client =  bullet_client.BulletClient(connection_mode=p1.DIRECT)
    self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_GUI,0)
    self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_Y_AXIS_UP,1)

    # load ground plane
    self._pybullet_client.setAdditionalSearchPath(URDF_DIR)
    z2y = self._pybullet_client.getQuaternionFromEuler([-math.pi*0.5,0,0])
    self._planeId = self._pybullet_client.loadURDF("plane_implicit.urdf",[0,0,0],z2y, useMaximalCoordinates=True)

    # set simulation environment parameters
    self._pybullet_client.setGravity(0,-9.8,0)

    self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=10)
    self._pybullet_client.changeDynamics(self._planeId, linkIndex=-1, lateralFriction=0.9)

    self._pybullet_client.setTimeStep(1.0/60.0)
    self._pybullet_client.setPhysicsEngineParameter(numSubSteps=1)

  def add_character(self, name):
    assert(name not in self.characters.keys())
    char_id = self._new_character()
    self.characters[name] = char_id
    return name

  def _new_character(self):

    if self._model == "humanoid3d":
      kin_model = self._pybullet_client.loadURDF(
        "humanoid/humanoid.urdf", [0,0.889540259,0],
        useFixedBase=True, flags=self._pybullet_client.URDF_MAINTAIN_LINK_ORDER)
    elif self._model == "atlas":
      kin_model = self._pybullet_client.loadURDF(
        "atlas/atlas.urdf", [0,0,0],
        useFixedBase=True, flags=self._pybullet_client.URDF_MAINTAIN_LINK_ORDER)
    else:
      raise NotImplementedError

    self._pybullet_client.changeDynamics(kin_model, -1, linearDamping=0, angularDamping=0)

    # set kinematic character dymanic, collision and vision property
    kin_act_state = (self._pybullet_client.ACTIVATION_STATE_SLEEP
                    + self._pybullet_client.ACTIVATION_STATE_ENABLE_SLEEPING
                    + self._pybullet_client.ACTIVATION_STATE_DISABLE_WAKEUP)
    self._pybullet_client.setCollisionFilterGroupMask(kin_model,-1,collisionFilterGroup=0,collisionFilterMask=0)
    self._pybullet_client.changeDynamics(kin_model,-1,activationState=kin_act_state)
    for j in range (self._pybullet_client.getNumJoints(kin_model)):
      self._pybullet_client.setCollisionFilterGroupMask(kin_model,j,collisionFilterGroup=0,collisionFilterMask=0)
      self._pybullet_client.changeDynamics(kin_model,j,activationState=kin_act_state)

    return kin_model

  def set_pose(self, char_name, pose, vel):
    """ Set character state in physics engine
      Inputs:
        pose   np.array of float, self._skeleton.pos_dim, position of base and
               orintation of joints, represented in local frame
        vel    np.array of float, self._skeleton.vel_dim, velocity of base and
               angular velocity of joints, represented in local frame

        phys_model  pybullet model unique Id, self._sim_model or self._kin_model
    """
    assert(char_name in self.characters.keys())
    phys_model = self.characters[char_name]
    s = self._skeleton
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

  def camera_follow(self, char_name, dis=None, yaw=None, pitch=None, pos=None):
    assert(char_name in self.characters.keys())
    phys_model = self.characters[char_name]

    _pos = np.array(self._get_base_pose(phys_model))
    _pos[1] = 1
    _pos[0] += 0.75

    cam_info = self._pybullet_client.getDebugVisualizerCamera()
    _yaw, _pitch, _dis = cam_info[8], cam_info[9], cam_info[10]
    dis = _dis if dis is None else dis
    yaw = _yaw if yaw is None else yaw
    pitch = _pitch if pitch is None else pitch
    pos = _pos if pos is None else pos
    self._pybullet_client.resetDebugVisualizerCamera(dis, yaw, pitch, pos)

  def _get_base_pose(self, phys_model):
    pos, orn = self._pybullet_client.getBasePositionAndOrientation(phys_model)
    return pos

  def _get_joint_pose(self, phys_model, jid):
    info = self._pybullet_client.getLinkState(phys_model, jid)
    pos = info[4]
    orn = info[5]
    return pos, orn
