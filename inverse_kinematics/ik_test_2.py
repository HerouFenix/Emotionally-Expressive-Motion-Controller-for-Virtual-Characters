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

kukaId = p.loadURDF("humanoid.urdf", [0,0.889540259, 0], flags=p.URDF_MAINTAIN_LINK_ORDER)
kukaEndEffectorIndex = 14

# Good ones: 8 - chest ; 10 - rshoulder
numJoints = p.getNumJoints(kukaId)
print(numJoints)
if (numJoints != 15):
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

ls = p.getLinkState(kukaId, kukaEndEffectorIndex)

i=0
while 1:
  i+=1
  t = t + 0.01

  for i in range(1):
    jointPoses = [-0.003271644330369293, -0.5266238886159375, 0.19250226350017366, 0.0, 0.0, 0.0, -0.5611316664634671, 0.023342605279165433, 1.159026094696599, 1.5155085326378472, 0.16585315274568407, 0.40904315564873656, -0.5620610019329436, 0.05739550254366494, 0.005279046213221956, -0.11618085378221293, -0.09071206681949875, -0.216116408, -0.037861925106024874, 0.18617740168938335, 0.3002627385428372, 0.06377706810167465, 0.23012967715817373, 0.538600147884017, -0.133930724, 0.05227730922639954, 0.07652195777290266, -0.015819457111757966]

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

               

    #p.resetBasePositionAndOrientation(kukaId, [0,0,0], [-0.7071067811865475, 0.0, 0.0, 0.7071067811865476])

    for j in range(numJoints):
      jtype = p.getJointInfo(kukaId,j)[2]
      if jtype == 0: #R/L Knee ; R/L Elbow
        joint_name = p.getJointInfo(kukaId,j)[1].decode("utf-8")
        orn = [jointPoses[mapping[joint_name][0]]]
        p.resetJointStateMultiDof(kukaId, j, orn)
      elif jtype == 2: #Chest ; Neck ; R/L Hip ; R/L Ankle ; R/L Shoulder
        joint_name = p.getJointInfo(kukaId,j)[1].decode("utf-8")
        
        orn = [jointPoses[mapping[joint_name][0]], jointPoses[mapping[joint_name][1]], jointPoses[mapping[joint_name][2]]] 
        orn = p.getQuaternionFromEuler(orn)

        p.resetJointStateMultiDof(kukaId, j, orn)

  print()

p.disconnect()

