frame_index = { 

        }

frame_offsets = { 

        }

mappings = { 
        "Hips":         ['2','1','-0'],
        "Spine":        ['1', '-0', '2'],
        "Spine1":        ['1','2','0'],
        "Neck":         ['1','2','0'],
        "Head":         ['1','2','0'],

        "RightUpLeg":   ['1','-0','2'],
        "RightLeg":   ['1','0','-2'],
        "RightFoot":       ['1','0','-2'],
        "RightToeBase": ['1','0','-2'],

        "RightShoulder":   ['1','2','0'],
        "RightArm":   ['0','-1','-2'],
        "RightForeArm":   ['1','0','-2'],
        "RightHand": ['1','0','-2'],

        "LeftUpLeg":   ['1','-0','2'],
        "LeftLeg":   ['-1','0','2'],
        "LeftFoot":       ['-1','0','2'],
        "LeftToeBase": ['-1','0','2'],

        "LeftShoulder":   ['1','2','0'],
        "LeftArm":   ['0','1','2'],
        "LeftForeArm": ['1','0','2'],
        "LeftHand": ['1','0','2'],
        }

order = []

frames = []

frame_time = 0.0333333

import numpy as np
import math
 
with open("walk_sad_og.bvh",'r') as f:
   line = f.readline()
   current_index = -1
   while line != "":
        if("MOTION" in line):
           break
        
        if("CHANNELS" in line):
            channel = line.strip()
            channel = channel.split(" ")
            current_index += int(channel[1])

        if("JOINT" in line or "ROOT" in line):
            joint = line.replace("\n","").replace("JOINT","").replace("ROOT","").strip()
            line = f.readline()
            line = f.readline()
            line = line.strip().split(" ")

            frame_offsets[joint] = [float(line[3]), float(line[1]), float(line[2])]
            #print("Joint OFFSET: " + joint + " | " + str(frame_offsets[joint]))
                
            line = f.readline().strip().split(" ")

            frame_index[joint] = [*range(current_index+1, current_index + 1 + int(line[1]), 1)]
            current_index += int(line[1])
            print("Joint INDEX: " + joint + " | " + str(frame_index[joint]))

            order.append(joint)

        line = f.readline()

print(order)
    
with open("walk_sad_og.bvh",'r') as f:
   line = f.readline()
   found = False

   while line != "":
        if(found):
            frame = line.split(" ")
            new_frame = []

            for joint in order:

                new_data = [float(frame[frame_index[joint][0]]),float(frame[frame_index[joint][1]]),float(frame[frame_index[joint][2]]), float(frame[frame_index[joint][3]]), float(frame[frame_index[joint][4]]), float(frame[frame_index[joint][5]])]

                if(joint == "LeftShoulder"):
                    for i in range(3):
                        mapping = abs(int(mappings[joint][i]))

                        if "-" in mappings[joint][i]:
                            new_data[mapping] = -float(frame[frame_index[joint][i]])
                        else:
                            new_data[mapping] = float(frame[frame_index[joint][i]])

            
                new_frame += new_data

            frames.append(new_frame)

        elif("Frame Time" in line):
            found = True

        line = f.readline()

print(np.array(frames))
np.savetxt("converted_file", np.array(frames))