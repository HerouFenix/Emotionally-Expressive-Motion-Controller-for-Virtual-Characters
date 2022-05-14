mappings = { 
        "Hips":"hip",
        "Spine1":"chest",
        "Neck":"neck",
        "RightUpLeg":"right hip",
        "RightLeg":"right knee",
        "RightFoot":"right ankle",
        "RightShoulder":"right shoulder",
        "RightArm":"right elbow",
        "LeftUpLeg":"left hip",
        "LeftLeg":"left knee",
        "LeftFoot":"left ankle",
        "LeftShoulder":"left shoulder",
        "LeftArm":"left elbow",
        "LeftHand": "",
        "RightHand": ""
        }

frame_index = { 
        "Hips":         [],
        "Spine1":        [],
        "Neck":         [],
        "RightUpLeg":   [],
        "RightLeg":   [],
        "RightFoot":       [],
        "RightShoulder":   [],
        "RightArm":   [],
        "LeftUpLeg":   [],
        "LeftLeg":   [],
        "LeftFoot":       [],
        "LeftShoulder":   [],
        "LeftArm":   [],
        "LeftHand": [],
        "RightHand": []
        }

frame_offsets = { 
        "Hips":         [],
        "Spine1":        [],
        "Neck":         [],
        "RightUpLeg":   [],
        "RightLeg":   [],
        "RightFoot":       [],
        "RightShoulder":   [],
        "RightArm":   [],
        "LeftUpLeg":   [],
        "LeftLeg":   [],
        "LeftFoot":       [],
        "LeftShoulder":   [],
        "LeftArm":   [],
        "LeftHand": [],
        "RightHand": []
        }

frames = []

frame_time = 0.0333333

import numpy as np
import math
 
def get_quaternion_from_euler(roll, pitch, yaw):
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

with open("walk_sad_converted.bvh",'r') as f:
   line = f.readline()
   current_index = -1
   while line != "":
        if("MOTION" in line):
           break
        
        if("CHANNELS" in line):
            channel = line.strip()
            channel = channel.split(" ")
            current_index += int(channel[1])

        if("JOINT" in line):
            joint = line.replace("\n","").replace("JOINT","").strip()
            if(joint in mappings.keys()):
                line = f.readline()
                line = f.readline().strip().split(" ")
                frame_offsets[joint] = [float(line[3]), float(line[1]), float(line[2])]
                #print("Joint OFFSET: " + joint + " | " + str(frame_offsets[joint]))
                
                line = f.readline().strip().split(" ")

                frame_index[joint] = [*range(current_index+1, current_index + 1 + int(line[1]), 1)]
                current_index += int(line[1])
                print("Joint INDEX: " + joint + " | " + str(frame_index[joint]))

        line = f.readline()
       

with open("walk_sad_converted.bvh",'r') as f:
   line = f.readline()
   found = False

   while line != "":
        if(found):
            frame = line.split(" ")

            new_frame = [frame_time]

            #Hips
            new_frame.append(float(frame[8]) * 0.056444)
            new_frame.append(float(frame[6]) * 0.056444)
            new_frame.append(float(frame[7]) * 0.056444)
            
            new_frame += get_quaternion_from_euler(float(frame[frame_index["Hips"][5]]), float(frame[frame_index["Hips"][3]]), float(frame[frame_index["Hips"][4]]))

            #chest            
            new_frame += get_quaternion_from_euler(float(frame[frame_index["Spine1"][5]]), float(frame[frame_index["Spine1"][3]]), float(frame[frame_index["Spine1"][4]]))

            #neck            
            new_frame += get_quaternion_from_euler(float(frame[frame_index["Neck"][5]]), float(frame[frame_index["Neck"][3]]), float(frame[frame_index["Neck"][4]]))

            #right hip            
            new_frame += get_quaternion_from_euler(float(frame[frame_index["RightUpLeg"][5]]), float(frame[frame_index["RightUpLeg"][3]]), float(frame[frame_index["RightUpLeg"][4]]))

            #right knee    
            pos_joint = np.array([float(frame[frame_index["RightLeg"][0]]) * 0.056444, float(frame[frame_index["RightLeg"][1]]) * 0.056444, float(frame[frame_index["RightLeg"][2]]) * 0.056444])
            pos_joint = pos_joint / np.linalg.norm(pos_joint)

            pos_child = np.array([float(frame[frame_index["RightFoot"][0]]) * 0.056444, float(frame[frame_index["RightFoot"][1]]) * 0.056444, float(frame[frame_index["RightFoot"][2]]) * 0.056444])
            pos_child = pos_child / np.linalg.norm(pos_child)

            angle = math.acos(np.dot(pos_joint, pos_child))
            new_frame.append(angle)

            #right ankle           
            new_frame += get_quaternion_from_euler(float(frame[frame_index["RightFoot"][5]]), float(frame[frame_index["RightFoot"][3]]), float(frame[frame_index["RightFoot"][4]]))

            #right shoulder           
            new_frame += get_quaternion_from_euler(float(frame[frame_index["RightShoulder"][5]]), float(frame[frame_index["RightShoulder"][3]]), float(frame[frame_index["RightShoulder"][4]]))

            #right elbow        
            pos_joint = np.array([float(frame[frame_index["RightArm"][0]]), float(frame[frame_index["RightArm"][1]]), float(frame[frame_index["RightArm"][2]])])
            pos_joint = pos_joint / np.linalg.norm(pos_joint)

            pos_child = np.array([float(frame[frame_index["RightHand"][0]]), float(frame[frame_index["RightHand"][1]]), float(frame[frame_index["RightHand"][2]])])
            pos_child = pos_child / np.linalg.norm(pos_child)

            angle = math.acos(np.dot(pos_joint, pos_child))
            new_frame.append(angle)   

            #left hip            
            new_frame += get_quaternion_from_euler(float(frame[frame_index["LeftUpLeg"][5]]), float(frame[frame_index["LeftUpLeg"][3]]), float(frame[frame_index["LeftUpLeg"][4]]))

            #left knee            
            pos_joint = np.array([float(frame[frame_index["LeftLeg"][0]]) * 0.056444, float(frame[frame_index["LeftLeg"][1]]) * 0.056444, float(frame[frame_index["LeftLeg"][2]]) * 0.056444])
            pos_joint = pos_joint / np.linalg.norm(pos_joint)

            pos_child = np.array([float(frame[frame_index["LeftFoot"][0]]) * 0.056444, float(frame[frame_index["LeftFoot"][1]]) * 0.056444, float(frame[frame_index["LeftFoot"][2]]) * 0.056444])
            pos_child = pos_child / np.linalg.norm(pos_child)

            angle = math.acos(np.dot(pos_joint, pos_child))
            new_frame.append(angle)   

            #left ankle           
            new_frame += get_quaternion_from_euler(float(frame[frame_index["LeftFoot"][5]]), float(frame[frame_index["LeftFoot"][3]]), float(frame[frame_index["LeftFoot"][4]]))

            #left shoulder           
            new_frame += get_quaternion_from_euler(float(frame[frame_index["LeftShoulder"][5]]), float(frame[frame_index["LeftShoulder"][3]]), float(frame[frame_index["LeftShoulder"][4]]))

            #left elbow           
            pos_joint = np.array([float(frame[frame_index["LeftArm"][0]]) * 0.056444, float(frame[frame_index["LeftArm"][1]]) * 0.056444, float(frame[frame_index["LeftArm"][2]]) * 0.056444])
            pos_joint = pos_joint / np.linalg.norm(pos_joint)

            pos_child = np.array([float(frame[frame_index["LeftHand"][0]]) * 0.056444, float(frame[frame_index["LeftHand"][1]]) * 0.056444, float(frame[frame_index["LeftHand"][2]]) * 0.056444])
            pos_child = pos_child / np.linalg.norm(pos_child)

            angle = math.acos(np.dot(pos_joint, pos_child))
            new_frame.append(angle)  


            frames.append(new_frame)

        elif("Frame Time" in line):
            found = True

        line = f.readline()
        

import json 

with open("walk_sad_converted.txt",'w') as f:
   f.write(json.dumps({"Loop": "none", "Frames": frames}, indent=4))