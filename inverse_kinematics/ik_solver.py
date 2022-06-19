import pybullet as p
import pybullet_utils.bullet_client as bc
import time
import math
from datetime import datetime

Z2Y = p.getQuaternionFromEuler([-math.pi*0.5,0,0])

class IKSolver():
    def __init__(self):
        # Setup PyBullet
        self.physicsClient = bc.BulletClient(connection_mode=p.DIRECT)
        #self.clid = self.physicsClient.connect(self.physicsClient.SHARED_MEMORY)

        self.physicsClient.setGravity(0, 0, 0)
        self.physicsClient.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP,1)
        self.physicsClient.setRealTimeSimulation(0)

        # Setup Character
        self.charID = self.physicsClient.loadURDF("../inverse_kinematics/humanoid_3.urdf", [0.5, 0.889540259, 0])
        #self.charID = self.physicsClient.loadURDF("../inverse_kinematics/humanoid_2.urdf", [0, 0, 0])
        #self.physicsClient.resetBasePositionAndOrientation(self.charID, [0, 0, 0], Z2Y)

        self.numJoints = self.physicsClient.getNumJoints(self.charID)

        self.ll = []
        self.ul = []
        self.jd = [0.1] * self.numJoints

        for i in range(self.numJoints):
            #if(self.physicsClient.getJointInfo(kukaId, i)[2] == 0):
            self.ll.append(self.physicsClient.getJointInfo(self.charID, i)[8])
            self.ul.append(self.physicsClient.getJointInfo(self.charID, i)[9])

        #_link_name_to_index = {self.physicsClient.getBodyInfo(self.charID)[0].decode('UTF-8'):-1,}
        #
        #for _id in range(self.physicsClient.getNumJoints(self.charID)):
        #    _name = self.physicsClient.getJointInfo(self.charID, _id)[12].decode('UTF-8')
        #    _link_name_to_index[_name] = _id
        #print(_link_name_to_index)

    def __del__(self):
        self.physicsClient.disconnect()

    def quaternion_multiply(self, Q0):
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
        w0 = Z2Y[3]
        x0 = Z2Y[0]
        y0 = Z2Y[1]
        z0 = Z2Y[2]
        
        # Computer the product of the two quaternions, term by term
        Q0Q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
        Q0Q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
        Q0Q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
        Q0Q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
        
        # Create a 4 element array containing the final quaternion
        final_quaternion = [Q0Q1_x, Q0Q1_y, Q0Q1_z, Q0Q1_w]
        
        # Return a 4 element arra
        return final_quaternion


   
    def euler_from_quaternion(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians
   
    def updatePose(self, pose):
        self.physicsClient.resetBasePositionAndOrientation(self.charID, [pose[0], pose[1], pose[2]], [pose[4], pose[5], pose[6], pose[3]])

        chest_rotation = self.physicsClient.getEulerFromQuaternion([pose[8],pose[9],pose[10],pose[7]])
        neck_rotation = self.physicsClient.getEulerFromQuaternion([pose[12],pose[13],pose[14],pose[11]])
        right_hip_rotation = self.physicsClient.getEulerFromQuaternion([pose[16],pose[17],pose[18],pose[15]])
        right_knee_rotation = pose[19]
        right_ankle_rotation = self.physicsClient.getEulerFromQuaternion([pose[21],pose[22],pose[23],pose[20]])
        right_shoulder_rotation = self.physicsClient.getEulerFromQuaternion([pose[25],pose[26],pose[27],pose[24]])
        right_elbow_rotation = pose[28]
        left_hip_rotation = self.physicsClient.getEulerFromQuaternion([pose[30],pose[31],pose[32],pose[29]])
        left_knee_rotation = pose[33]
        left_ankle_rotation = self.physicsClient.getEulerFromQuaternion([pose[35],pose[36],pose[37],pose[34]])
        left_shoulder_rotation = self.physicsClient.getEulerFromQuaternion([pose[39],pose[40],pose[41],pose[38]])
        left_elbow_rotation = pose[42]

        pose = [
                chest_rotation[2],
                chest_rotation[1],
                chest_rotation[0],

                neck_rotation[2],
                neck_rotation[1],
                neck_rotation[0],

                right_shoulder_rotation[2],
                right_shoulder_rotation[1],
                right_shoulder_rotation[0],

                right_elbow_rotation,

                left_shoulder_rotation[2],
                left_shoulder_rotation[1],
                left_shoulder_rotation[0],

                left_elbow_rotation,

                right_hip_rotation[2],
                right_hip_rotation[1],
                right_hip_rotation[0],

                right_knee_rotation,

                right_ankle_rotation[2],
                right_ankle_rotation[1],
                right_ankle_rotation[0],

                left_hip_rotation[2],
                left_hip_rotation[1],
                left_hip_rotation[0],

                left_knee_rotation,

                left_ankle_rotation[2],
                left_ankle_rotation[1],
                left_ankle_rotation[0],
            ]
            
        counter = 0
        for j in range(self.numJoints):

            if(self.physicsClient.getJointInfo(self.charID,j)[2] == 0):
                #print(self.physicsClient.getJointInfo(self.charID, j)[1])

                pos = pose[counter]

                self.physicsClient.resetJointState(self.charID, j, pos)

                counter += 1

    def adjustBase(self, newHeight):
        poses = []
        orn = []
        """
        25 - rknee
        29 - rankle
        34 - lknee
        38 - lankle
        """
        index = [25, 29, 34, 38]
        for j in index:
            linkState = self.physicsClient.getLinkState(self.charID, j)
            poses.append(linkState[4])
            orn.append(linkState[5])

        basePosAndOrn = self.physicsClient.getBasePositionAndOrientation(self.charID)
        pos = [basePosAndOrn[0][0], newHeight, basePosAndOrn[0][2]]
        self.physicsClient.resetBasePositionAndOrientation(self.charID, pos, basePosAndOrn[1])

        return self.calculateKinematicSolution2(index, poses)

    def getLinkState(self, link_id):
        return self.physicsClient.getLinkState(self.charID, link_id)

    def calculateKinematicSolution(self, endEffectorIndex, desiredPosition, desiredOrientation = None, jd=[]):
        if(len(jd) == 0):
            jd = self.jd

        if(desiredOrientation != None):
            jointPoses = self.physicsClient.calculateInverseKinematics(self.charID, endEffectorIndex, desiredPosition, desiredOrientation, jointDamping=jd)
        else:
            jointPoses = self.physicsClient.calculateInverseKinematics(self.charID, endEffectorIndex, desiredPosition, jointDamping=jd
            )

        counter = 0
        for j in range(self.numJoints):
            if(self.physicsClient.getJointInfo(self.charID,j)[2] == 0):
                self.physicsClient.resetJointState(self.charID, j, jointPoses[counter])
                counter += 1
        
        return jointPoses

    def calculateKinematicSolution2(self, endEffectorIndices, desiredPositions, jd = []):
        if(len(jd) == 0):
            jd = self.jd

        jointPoses = self.physicsClient.calculateInverseKinematics2(self.charID, endEffectorIndices, desiredPositions, jointDamping = jd)
        
        counter = 0
        for j in range(self.numJoints):
            if(self.physicsClient.getJointInfo(self.charID,j)[2] == 0):
                self.physicsClient.resetJointState(self.charID, j, jointPoses[counter])
                counter += 1

        return jointPoses

    def accurateCalculateInverseKinematics(self, endEffectorId, targetPos, targetOrn, threshold, maxIter):
        closeEnough = False
        iter = 0
        dist2 = 1e30
        while (not closeEnough and iter < maxIter):
            jointPoses = self.physicsClient.calculateInverseKinematics(self.charID, endEffectorId, targetPos, targetOrn)
            counter = 0
            for j in range(self.numJoints):
                if(self.physicsClient.getJointInfo(self.charID,j)[2] == 0):
                    self.physicsClient.resetJointState(self.charID, j, jointPoses[counter])
                    counter += 1

            ls = self.physicsClient.getLinkState(self.charID, endEffectorId)
            newPos = ls[4]
            diff = [targetPos[0] - newPos[0], targetPos[1] - newPos[1], targetPos[2] - newPos[2]]
            dist2 = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            closeEnough = (dist2 < threshold)
            iter = iter + 1
        print ("Num iter: "+str(iter) + " Threshold: "+str(dist2))
        return jointPoses

    def calculateAccurateKinematicSolution(self, endEffectorIndex, desiredPosition, desiredOrientation):
        jointPoses = self.accurateCalculateInverseKinematics(endEffectorIndex, desiredPosition, desiredOrientation, 0.00001, 100)

        counter = 0
        for j in range(self.numJoints):
            if(self.physicsClient.getJointInfo(self.charID,j)[2] == 0):
                self.physicsClient.resetJointState(self.charID, j, jointPoses[counter])
                counter += 1

        return jointPoses

