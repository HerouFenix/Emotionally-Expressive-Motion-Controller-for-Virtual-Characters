import math
from ntpath import join
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

  def quaternion_multiply(self, Q0, baseOld, baseNew):
    """
    Multiplies two quaternions.
 
    Input
    :param Q0: A 4 element array containing the first quaternion (q01,q11,q21,q31) 
    :param Q1: A 4 element array containing the second quaternion (q02,q12,q22,q32) 
 
    Output
    :return: A 4 element array containing the final quaternion (q03,q13,q23,q33) 
 
    """
    # Extract the values from Q0
    w0 = Q0[3]
    x0 = Q0[0]
    y0 = Q0[1]
    z0 = Q0[2]
     
    # Extract the values from Q1
    w1 = baseOld[3]
    x1 = -baseOld[0]
    y1 = -baseOld[1]
    z1 = -baseOld[2]

    # Extract the values from Q1
    w2 = baseNew[3]
    x2 = baseNew[0]
    y2 = baseNew[1]
    z2 = baseNew[2]
     
    # Computer the product of the two quaternions, term by term
    Q0Q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    Q0Q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    Q0Q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    Q0Q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1

    Q0Q2_x = Q0Q1_w * x2 + Q0Q1_x * w2 + Q0Q1_y * z2 - Q0Q1_z * y2
    Q0Q2_y = Q0Q1_w * y2 - Q0Q1_x * z2 + Q0Q1_y * w2 + Q0Q1_z * x2
    Q0Q2_z = Q0Q1_w * z2 + Q0Q1_x * y2 - Q0Q1_y * x2 + Q0Q1_z * w2
    Q0Q2_w = Q0Q1_w * w2 - Q0Q1_x * x2 - Q0Q1_y * y2 - Q0Q1_z * z2
     
    # Create a 4 element array containing the final quaternion
    final_quaternion = [Q0Q2_x, Q0Q2_y, Q0Q2_z, Q0Q2_w]
     
    # Return a 4 element arra
    return final_quaternion

  def quaternion_multiply(self,Q0):
    """
    Multiplies two quaternions.
 
    Input
    :param Q0: A 4 element array containing the first quaternion (q01,q11,q21,q31) 
    :param Q1: A 4 element array containing the second quaternion (q02,q12,q22,q32) 
 
    Output
    :return: A 4 element array containing the final quaternion (q03,q13,q23,q33) 
 
    """
    # Extract the values from Q0
    w1 = Q0[3]
    x1 = Q0[0]
    y1 = Q0[1]
    z1 = Q0[2]
     
    # Extract the values from Q1
    y2z = self._pybullet_client.getQuaternionFromEuler([-math.pi*0.5,0,0]) 
    w0 = y2z[3]
    x0 = y2z[0]
    y0 = y2z[1]
    z0 = y2z[2]
     
    # Computer the product of the two quaternions, term by term
    Q0Q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    Q0Q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    Q0Q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    Q0Q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
     
    # Create a 4 element array containing the final quaternion
    final_quaternion = [Q0Q1_x, Q0Q1_y, Q0Q1_z, Q0Q1_w]
     
    # Return a 4 element arra
    return final_quaternion

  def get_quaternion_from_euler(self, roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.
    
    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.
  
    Output
      :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  
    return [qx, qy, qz, qw]


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

    #pose = [ 4.98910587e-01,  8.61965996e-01,  6.76430121e-03,  1.00000000e+00,
    #0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  9.99938012e-01,
    #0.00000000e+00, -1.11054544e-02, -8.00608165e-04,  1.00000000e+00,
    #0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  9.97290026e-01,
    #0.00000000e+00, -5.81173275e-02, -4.51107603e-02, -2.16116408e-01,
    #9.84030931e-01, -3.25363635e-02,  8.90732054e-02,  1.50633584e-01,
    #9.39676420e-01, -1.98687524e-01,  2.35818497e-01,  1.48057864e-01,
    #5.71147477e-01,  9.58070112e-01,  0.00000000e+00,  1.19043367e-01,
    #2.60634491e-01, -1.33930724e-01,  9.98887637e-01,  2.64181872e-02,
    #3.80308091e-02, -8.90085879e-03,  9.74355800e-01,  1.03421713e-01,
    #1.38056048e-01, -1.44482701e-01,  1.50277310e-01,]

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
        #print(self._pybullet_client.getJointInfo(phys_model, i)[1])
        #print(self._pybullet_client.getJointStateMultiDof(phys_model, i)[0])
        #print()
        
      elif jtype is JointType.SPHERE:
        orn_wxyz = pose[p_off : p_off+4]
        orn = [orn_wxyz[1], orn_wxyz[2], orn_wxyz[3], orn_wxyz[0]]
        omg = vel[p_off : p_off+3]
        self._pybullet_client.resetJointStateMultiDof(phys_model, i, orn, omg)

        #print(self._pybullet_client.getJointInfo(phys_model, i)[1])
        #print(self._pybullet_client.getJointStateMultiDof(phys_model, i)[0])
        #print()

    #TODO: Pass current pose to IKSolver ; 
    #TODO: Pass desired position to IKSolver
    #TODO: Get jointPoses from IKSolver
    #jointPoses = IKSolver.computeInverseKinematics(current_pose, desired_positions)
    """
    jointPoses = [0.4650167160596638, -0.276508758220416, 0.19514429851526932, 0.0, 0.0, 0.0, -0.5209967176723092, 0.4877744736788875, 0.18807214304180003, 1.005843252835603, 1.6563578605484204, -1.311992045463749, -0.8451394219408266, 0.005276910786724907, 0.005279046213221956, -0.11618085378221293, -0.09071206681949875, -0.216116408, -0.037861925106024874, 0.18617740168938335, 0.3002627385428372, 0.06377706810167465, 0.23012967715817373, 0.538600147884017, -0.133930724, 0.05227730922639954, 0.07652195777290266, -0.015819457111757966]

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

    #curPos, curOrn = self._pybullet_client.getBasePositionAndOrientation(phys_model)
    #self._pybullet_client.resetBasePositionAndOrientation(phys_model, curPos, self.quaternion_multiply(curOrn))
    #print(self._pybullet_client.getBasePositionAndOrientation(phys_model)[1])

    
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
        print(joint_name)
        print(mapping[joint_name])
        
        orn = [jointPoses[mapping[joint_name][0]], jointPoses[mapping[joint_name][1]], jointPoses[mapping[joint_name][2]]] 
        orn = self._pybullet_client.getQuaternionFromEuler(orn)

        self._pybullet_client.resetJointStateMultiDof(phys_model, i, orn)
    """



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
