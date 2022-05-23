from ntpath import join
import pybullet as p
import time
import math
from datetime import datetime


Z2Y = p.getQuaternionFromEuler([-math.pi*0.5,0,0])

class IKSolver():
    def __init__(self):
        # Setup PyBullet
        self.clid = p.connect(p.SHARED_MEMORY)
        if (self.clid < 0):
            p.connect(p.GUI)
            #p.connect(p.SHARED_MEMORY_GUI)

        p.setGravity(0, 0, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP,1)
        p.setRealTimeSimulation(0)

        # Setup Character
        self.charID = p.loadURDF("humanoid_2.urdf", [0, 0, 0])
        p.resetBasePositionAndOrientation(self.charID, [0, 0, 0], Z2Y)

        self.numJoints = p.getNumJoints(self.charID)

        self.ll = []
        self.ul = []
        self.jd = [1.0] * self.numJoints

        for i in range(self.numJoints):
            #if(p.getJointInfo(kukaId, i)[2] == 0):
            self.ll.append(p.getJointInfo(self.charID, i)[8])
            self.ul.append(p.getJointInfo(self.charID, i)[9])

    def __del__(self):
        p.disconnect(self.clid)

        
    def accurateCalculateInverseKinematics(self, endEffectorId, targetPos, threshold, maxIter):
        closeEnough = False
        iter = 0
        dist2 = 1e30
        while (not closeEnough and iter < maxIter):
            jointPoses = p.calculateInverseKinematics(self.charID, endEffectorId, targetPos, lowerLimits=self.ll, upperLimits=self.ul, jointDamping=self.jd)
            counter = 0
            for j in range(self.numJoints):
                if(p.getJointInfo(self.charID,j)[2] == 0):
                    p.resetJointState(self.charID, j, jointPoses[counter])
                    counter += 1

            ls = p.getLinkState(self.charID, endEffectorId)
            newPos = ls[4]
            diff = [targetPos[0] - newPos[0], targetPos[1] - newPos[1], targetPos[2] - newPos[2]]
            dist2 = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            closeEnough = (dist2 < threshold)
            iter = iter + 1
        print ("Num iter: "+str(iter) + " Threshold: "+str(dist2))
        return jointPoses

    def updatePose(self):
        pass

    def calculateKinematicSolution(self, desiredPositions, endEffectorIndices):
        jointPoses = p.calculateInverseKinematics2(self.charID, endEffectorIndices, desiredPositions, lowerLimits=self.ll, upperLimits=self.ul, jointDamping=self.jd)

        return jointPoses
