import pybullet as p
import time
import math
from datetime import datetime
import pybullet_data

clid = p.connect(p.SHARED_MEMORY)
if (clid < 0):
  p.connect(p.GUI)
  #p.connect(p.SHARED_MEMORY_GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

kukaId = p.loadURDF("humanoid_2.urdf", [0, 0, 0])
p.resetBasePositionAndOrientation(kukaId, [0, 0, 0], [0, 0, 0, 1])
kukaEndEffectorIndex = 28

# Good ones: 8 - chest ; 10 - rshoulder
numJoints = p.getNumJoints(kukaId)
print(numJoints)
if (numJoints != 32):
  exit()



p.setGravity(0, 0, 0)
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

ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
c = [p.addUserDebugParameter("rWrist x", -2, 2, ls[4][0]), p.addUserDebugParameter("rWrist y", -2, 2, ls[4][1]), p.addUserDebugParameter("rWrist z", -2, 2, ls[4][2])]

i=0
while 1:
  i+=1
  t = t + 0.01

  for i in range(1):
    ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
    print("Initial: " + str(ls[4]))

    pos = [p.readUserDebugParameter(c[0]), p.readUserDebugParameter(c[1]), p.readUserDebugParameter(c[2])]

    #end effector points down, not up (in case useOrientation==1)
    orn = ls[5]

    jointPoses = p.calculateInverseKinematics(kukaId, kukaEndEffectorIndex, pos)

    #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
    counter = 0
    for j in range(numJoints):
        if(p.getJointInfo(kukaId,j)[2] == 0):
            p.resetJointState(kukaId, j, jointPoses[counter])
            counter += 1

  ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
  print("Final: " + str(ls[4]))
  print()

p.disconnect()

"""
{'base_link': -1, 'base': 0, 'root': 1, 'root_chest_link1': 2, 'root_chest_link2': 3, 'chest': 4, 'chest_neck_link1': 5, 'chest_neck_link2': 6, 'neck': 7, 'chest_right_shoulder_link1': 8, 'chest_right_shoulder_link2': 9, 'right_shoulder': 10, 'right_elbow': 11, 'right_wrist': 12, 'chest_left_shoulder_link1': 13, 'chest_left_shoulder_link2': 14, 'left_shoulder': 15, 'left_elbow': 16, 'left_wrist': 17, 'root_right_hip_link1': 18, 'root_right_hip_link2': 19, 'right_hip': 20, 'right_knee': 21, 'right_knee_right_ankle_link1': 22, 'right_knee_right_ankle_link2': 23, 'right_ankle': 24, 'root_left_hip_link1': 25, 'root_left_hip_link2': 26, 'left_hip': 27, 'left_knee': 28, 'left_knee_left_ankle_link1': 29, 'left_knee_left_ankle_link2': 30, 'left_ankle': 31}
"""

"""
JOINTPOSE:
[b'root_chest_joint1', b'root_chest_joint2', b'root_chest_joint3', b'chest_neck_joint1', b'chest_neck_joint2', b'chest_neck_joint3', b'chest_right_shoulder_joint1', b'chest_right_shoulder_joint2', b'chest_right_shoulder_joint3', b'right_elbow', b'chest_left_shoulder_joint1', b'chest_left_shoulder_joint2', b'chest_left_shoulder_joint3', b'left_elbow', b'root_right_hip_joint1', b'root_right_hip_joint2', b'root_right_hip_joint3', b'right_knee', b'right_knee_right_ankle_joint1', b'right_knee_right_ankle_joint2', b'right_knee_right_ankle_joint3', b'root_left_hip_joint1', b'root_left_hip_joint2', b'root_left_hip_joint3', b'left_knee', b'left_knee_left_ankle_joint1', b'left_knee_left_ankle_joint2', b'left_knee_left_ankle_joint3']
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