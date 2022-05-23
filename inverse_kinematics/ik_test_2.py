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

kukaId = p.loadURDF("humanoid.urdf", [0, 0, 0], flags=p.URDF_MAINTAIN_LINK_ORDER)
p.resetBasePositionAndOrientation(kukaId, [0, 0, 0], [0, 0, 0, 1])
kukaEndEffectorIndex = 14

# Good ones: 8 - chest ; 10 - rshoulder
numJoints = p.getNumJoints(kukaId)
print(numJoints)
if (numJoints != 15):
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

    #pos = [p.readUserDebugParameter(c[0]), p.readUserDebugParameter(c[1]), p.readUserDebugParameter(c[2])]

    #end effector points down, not up (in case useOrientation==1)
    orn = ls[5]

    jointPoses = p.calculateInverseKinematics(kukaId, kukaEndEffectorIndex, [0,0,0])

    #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
    #counter = 0
    #for j in range(numJoints):
    #    if(p.getJointInfo(kukaId,j)[2] == 0):
    #        p.resetJointState(kukaId, j, jointPoses[counter])
    #        counter += 1

  ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
  print("Final: " + str(ls[4]))

  #print("Joint Poses: " + str(jointPoses))
  print()

p.disconnect()

