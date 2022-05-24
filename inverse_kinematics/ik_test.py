import pybullet as p
import time
import math
from datetime import datetime
import pybullet_data
import numpy as np

clid = p.connect(p.SHARED_MEMORY)
if (clid < 0):
  p.connect(p.GUI)
  #p.connect(p.SHARED_MEMORY_GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

z2y = p.getQuaternionFromEuler([-math.pi*0.5,0,0]) 
kukaId = p.loadURDF("humanoid_2.urdf", [0,0.889540259, 0], z2y)
kukaEndEffectorIndexR = 12
kukaEndEffectorIndexL = 17

# Good ones: 8 - chest ; 10 - rshoulder
numJoints = p.getNumJoints(kukaId)
print(numJoints)
if (numJoints != 32):
  exit()



p.setGravity(0, 0, 0)
p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP,1)
t = 0.
prevPose = [0, 0, 0]
prevPose1 = [0, 0, 0]
hasPrevPose = 0
useNullSpace = 1

useOrientation = 1
#If we set useSimulation=0, it sets the arm pose to be the IK result directly without using dynamic control.
#This can be used to test the IK result accuracy.
useSimulation = 1
useRealTimeSimulation = 0
ikSolver = 0
p.setRealTimeSimulation(useRealTimeSimulation)
#trailDuration is duration (in seconds) after debug lines will be removed automatically
#use 0 for no-removal

"""
counter = 0
jointposes=[]
for i in range(numJoints):
    print(i)
    print(p.getJointInfo(kukaId, i)[1])
    print(p.getJointInfo(kukaId,i)[2])
    if(p.getJointInfo(kukaId,i)[2] == 0):
        jointposes.append(p.getJointInfo(kukaId, i)[1])
        counter += 1
    print()

print(counter)
print(len(jointposes))
"""

def quaternion_multiply(Q0):
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
    w0 = z2y[3]
    x0 = z2y[0]
    y0 = z2y[1]
    z0 = z2y[2]
     
    # Computer the product of the two quaternions, term by term
    Q0Q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    Q0Q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    Q0Q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    Q0Q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
     
    # Create a 4 element array containing the final quaternion
    final_quaternion = [Q0Q1_x, Q0Q1_y, Q0Q1_z, Q0Q1_w]
     
    # Return a 4 element arra
    return final_quaternion


#Pose is in W X Y Z ; PyBullet needs X Y Z W
pose = [ 4.98910587e-01,  8.61965996e-01,  6.76430121e-03,  1.00000000e+00,
0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  9.99938012e-01,
0.00000000e+00, -1.11054544e-02, -8.00608165e-04,  1.00000000e+00,
0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  9.97290026e-01,
0.00000000e+00, -5.81173275e-02, -4.51107603e-02, -2.16116408e-01,
9.84030931e-01, -3.25363635e-02,  8.90732054e-02,  1.50633584e-01,
9.39676420e-01, -1.98687524e-01,  2.35818497e-01,  1.48057864e-01,
5.71147477e-01,  9.58070112e-01,  0.00000000e+00,  1.19043367e-01,
2.60634491e-01, -1.33930724e-01,  9.98887637e-01,  2.64181872e-02,
 3.80308091e-02, -8.90085879e-03,  9.74355800e-01,  1.03421713e-01,
1.38056048e-01, -1.44482701e-01,  1.50277310e-01,]


p.resetBasePositionAndOrientation(kukaId, [pose[0], pose[1], pose[2]], quaternion_multiply([pose[4], pose[5], pose[6], pose[3]]))

chest_rotation = p.getEulerFromQuaternion([pose[8],pose[9],pose[10],pose[7]])
neck_rotation = p.getEulerFromQuaternion([pose[12],pose[13],pose[14],pose[11]])
right_hip_rotation = p.getEulerFromQuaternion([pose[16],pose[17],pose[18],pose[15]])
right_knee_rotation = pose[19]
right_ankle_rotation = p.getEulerFromQuaternion([pose[21],pose[22],pose[23],pose[20]])
right_shoulder_rotation = p.getEulerFromQuaternion([pose[25],pose[26],pose[27],pose[24]])
right_elbow_rotation = pose[28]
left_hip_rotation = p.getEulerFromQuaternion([pose[30],pose[31],pose[32],pose[29]])
left_knee_rotation = pose[33]
left_ankle_rotation = p.getEulerFromQuaternion([pose[35],pose[36],pose[37],pose[34]])
left_shoulder_rotation = p.getEulerFromQuaternion([pose[39],pose[40],pose[41],pose[38]])
left_elbow_rotation = pose[42]

pose = [
        chest_rotation[0],
        chest_rotation[1],
        chest_rotation[2],
        neck_rotation[0],
        neck_rotation[1],
        neck_rotation[2],
        right_shoulder_rotation[0],
        right_shoulder_rotation[1],
        right_shoulder_rotation[2],
        right_elbow_rotation,
        left_shoulder_rotation[0],
        left_shoulder_rotation[1],
        left_shoulder_rotation[2],
        left_elbow_rotation,
        right_hip_rotation[0],
        right_hip_rotation[1],
        right_hip_rotation[2],
        right_knee_rotation,
        right_ankle_rotation[0],
        right_ankle_rotation[1],
        right_ankle_rotation[2],
        left_hip_rotation[0],
        left_hip_rotation[1],
        left_hip_rotation[2],
        left_knee_rotation,
        left_ankle_rotation[0],
        left_ankle_rotation[1],
        left_ankle_rotation[2],
    ]
      
counter = 0
for j in range(numJoints):
      if(p.getJointInfo(kukaId,j)[2] == 0):
          print(p.getJointInfo(kukaId, j)[1])
          print(counter)
          p.resetJointState(kukaId, j, pose[counter])
          counter += 1


ll = []
ul = []
jd = []
for i in range(numJoints):
  #if(p.getJointInfo(kukaId, i)[2] == 0):
  ll.append(p.getJointInfo(kukaId, i)[8])
  ul.append(p.getJointInfo(kukaId, i)[9])


ls = p.getLinkState(kukaId, kukaEndEffectorIndexR)
c = [p.addUserDebugParameter("rWrist x", -2, 2, ls[4][0]), p.addUserDebugParameter("rWrist y", -2, 2, ls[4][1]), p.addUserDebugParameter("rWrist z", -2, 2, ls[4][2])]
orn_r = ls[5]


ls = p.getLinkState(kukaId, kukaEndEffectorIndexL)
c2 = [p.addUserDebugParameter("LWrist x", -2, 2, ls[4][0]), p.addUserDebugParameter("LWrist y", -2, 2, ls[4][1]), p.addUserDebugParameter("LWrist z", -2, 2, ls[4][2])]
orn_l = ls[5]

i=0
while 1:
  i+=1
  t = t + 0.01

  for i in range(1):
    #ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
    #print("Initial: " + str(ls[4]))

    pos_r = [p.readUserDebugParameter(c[0]), p.readUserDebugParameter(c[1]), p.readUserDebugParameter(c[2])]
    pos_l = [p.readUserDebugParameter(c2[0]), p.readUserDebugParameter(c2[1]), p.readUserDebugParameter(c2[2])]

    #end effector points down, not up (in case useOrientation==1)
    

    # NOTE: Theres an issue when using calcInvKin2 that has to do with us not being able to specify the orientations
    jointPoses = p.calculateInverseKinematics2(kukaId, [kukaEndEffectorIndexR, kukaEndEffectorIndexL], [pos_r, pos_l] ,lowerLimits=ll, upperLimits=ul, jointDamping=[0.01] * 32)
    #jointPoses = accurateCalculateInverseKinematics(numJoints, kukaId, kukaEndEffectorIndex, pos, 0.01, 100)
    
    #jointPoses = p.calculateInverseKinematics(kukaId, kukaEndEffectorIndexR, pos_r, orn_r, lowerLimits=ll, upperLimits=ul, jointDamping=[0.1] * 32)

    #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
    counter = 0
    for j in range(numJoints):
        if(p.getJointInfo(kukaId,j)[2] == 0):
            print(p.getJointInfo(kukaId,j)[1].decode("utf-8"))
            p.resetJointState(kukaId, j, jointPoses[counter])
            counter += 1
    """
    jointPoses = p.calculateInverseKinematics(kukaId, kukaEndEffectorIndexL, pos_l, orn_l, lowerLimits=ll, upperLimits=ul, jointDamping=[0.1] * 32)

    #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
    counter = 0
    for j in range(numJoints):
        if(p.getJointInfo(kukaId,j)[2] == 0):
            print(p.getJointInfo(kukaId,j)[1].decode("utf-8"))
            p.resetJointState(kukaId, j, jointPoses[counter])
            counter += 1
    """

    pose = []
    for j in range(numJoints):
        if(p.getJointInfo(kukaId,j)[2] == 0):
            print(p.getJointInfo(kukaId,j)[1].decode("utf-8"))
            pose.append(p.getJointState(kukaId, j)[0])
    


  #ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
  #print("Final: " + str(ls[4]))
  print("Poses: " + str(pose))
  print("Joint Poses: " + str(jointPoses))
  print("Root:" + str(p.getBasePositionAndOrientation(kukaId)[1]))
  print()

  time.sleep(1)

p.disconnect()

"""
{'base_link': -1, 'base': 0, 'root': 1, 'root_chest_link1': 2, 'root_chest_link2': 3, 'chest': 4, 'chest_neck_link1': 5, 'chest_neck_link2': 6, 'neck': 7, 'chest_right_shoulder_link1': 8, 'chest_right_shoulder_link2': 9, 'right_shoulder': 10, 'right_elbow': 11, 'right_wrist': 12, 'chest_left_shoulder_link1': 13, 'chest_left_shoulder_link2': 14, 'left_shoulder': 15, 'left_elbow': 16, 'left_wrist': 17, 'root_right_hip_link1': 18, 'root_right_hip_link2': 19, 'right_hip': 20, 'right_knee': 21, 'right_knee_right_ankle_link1': 22, 'right_knee_right_ankle_link2': 23, 'right_ankle': 24, 'root_left_hip_link1': 25, 'root_left_hip_link2': 26, 'left_hip': 27, 'left_knee': 28, 'left_knee_left_ankle_link1': 29, 'left_knee_left_ankle_link2': 30, 'left_ankle': 31}
"""

"""
JOINTPOSE:
[
  00-02 b'root_chest_joint1', b'root_chest_joint2', b'root_chest_joint3', 
  03-05 b'chest_neck_joint1', b'chest_neck_joint2', b'chest_neck_joint3', 
  06-08 b'chest_right_shoulder_joint1', b'chest_right_shoulder_joint2', b'chest_right_shoulder_joint3', 
  09    b'right_elbow', 
  10-12 b'chest_left_shoulder_joint1', b'chest_left_shoulder_joint2', b'chest_left_shoulder_joint3',
  13    b'left_elbow', 
  14-16 b'root_right_hip_joint1', b'root_right_hip_joint2', b'root_right_hip_joint3', 
  17    b'right_knee', 
  18-20 b'right_knee_right_ankle_joint1', b'right_knee_right_ankle_joint2', b'right_knee_right_ankle_joint3', 
  21-23 b'root_left_hip_joint1', b'root_left_hip_joint2', b'root_left_hip_joint3', 
  24    b'left_knee', 
  25-27 b'left_knee_left_ankle_joint1', b'left_knee_left_ankle_joint2', b'left_knee_left_ankle_joint3'
]

(-0.04673124029572879, -0.17368473610432916, 0.04654908533054814, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.480219482848072, -0.20724234344843553, -0.08050166649627483, -5.01473520474913e-15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
"""

"""
JOINTS:
0
b'base_base_link_rotation'
4

1
b'root'
4

2
b'root_chest_joint1'
0

3
b'root_chest_joint2'
0

4
b'root_chest_joint3'
0

5
b'chest_neck_joint1'
0

6
b'chest_neck_joint2'
0

7
b'chest_neck_joint3'
0

8
b'chest_right_shoulder_joint1'
0

9
b'chest_right_shoulder_joint2'
0

10
b'chest_right_shoulder_joint3'
0

11
b'right_elbow'
0

12
b'right_wrist'
4

13
b'chest_left_shoulder_joint1'
0

14
b'chest_left_shoulder_joint2'
0

15
b'chest_left_shoulder_joint3'
0

16
b'left_elbow'
0

17
b'left_wrist_joint'
4

18
b'root_right_hip_joint1'
0

19
b'root_right_hip_joint2'
0

20
b'root_right_hip_joint3'
0

21
b'right_knee'
0

22
b'right_knee_right_ankle_joint1'
0

23
b'right_knee_right_ankle_joint2'
0

24
b'right_knee_right_ankle_joint3'
0

25
b'root_left_hip_joint1'
0

26
b'root_left_hip_joint2'
0

27
b'root_left_hip_joint3'
0

28
b'left_knee'
0

29
b'left_knee_left_ankle_joint1'
0

30
b'left_knee_left_ankle_joint2'
0

31
b'left_knee_left_ankle_joint3'
0

"""