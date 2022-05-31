import math
import numpy as np
import os.path
import os

import pandas as pd


import pandas as pd

import xgboost as xgb


xgb.set_config(verbosity=0)
from scipy.optimize import minimize

#from gui_manager import GUIManager

ANGRY_LMA = [0.651203, 0.200424, 0.249959, 0.419565, 0.434269, 0.457952, 0.315414, 0.345453, 0.286151, 0.278876, 0.011726, 0.133865, 0.063344,
             0.988637, 0.097957, 0.126078, 0.03653, 0.989117, 0.655308, 1.164914, 0.471703, 0.920887, 1.978233, 1.310616, 2.329827, 0.943406, 1.841774, ]

# TODO: CHANGE T-POSE TO GET LOCAL FRAME INSTEAD OF CENTER OF MASS!!!!!
T_POSE = {
    'frame': [-0.16698190632853216, 0.8208478789814777, 0.1910868181792611, 0.9992503855746498, -0.0007243134215494251, -0.035510798615893846, -0.015398879200849844, 1.0, 0.0, 0.0, -0.0, 0.9987343710490267, 0.0001858002755274909, 1.2076377216405983e-05, 0.05029534190880983, 0.22476234873632903, 0.007103624462584861, 0.973989785667957, -0.02784461392281431, 0.01804881527261475, 0.0048504515100555685, 0.13429437369005343, 0.9903194749825479, -0.03476825830603958, 0.5367113370923698, -0.10618192214635794, -0.5577237662437339, 0.6241879048914117, 0.02954824366798683, 0.21699504864113023, -0.0004543469176178096, -0.9752027536120662, -0.043503238747886544, 0.03141429228324602, 0.0072404068297222774, -0.1344571744987444, -0.9896361284606645, -0.04989166243112469, 0.7368941237009464, 0.5344653215968344, 0.35230940537492045, 0.21728311795852154, 0.060142010491652034], 
    'root': [(-0.16698190569877625, 0.8208478689193726, 0.1910868138074875), (-0.0007243134314194322, -0.035510800778865814, -0.01539888046681881, 0.9992503523826599), [(0.001954074012308122, 0.0006167658321652191, 0.0001598828422912501), (-0.016494608265362715, -0.004274316497034636, -0.005580577097766774)]], 
    'chest': [(-0.15970230102539062, 1.0568866729736328, 0.1910032480955124), (-0.0007243134314194322, -0.035510800778865814, -0.01539888046681881, 0.9992503523826599), (6.376130714032577e-124, -7.970163267467595e-125, 6.376130664003356e-124)], 
    'neck': [(-0.152800515294075, 1.2806742191314697, 0.1909240037202835), (-0.0023235774133354425, -0.035420216619968414, 0.03488483652472496, 0.9987607598304749), (0.016168579070208654, 0.0002179010454178043, 0.00041141508245339057)], 
    'right_hip': [(-0.17300429940223694, 0.8210635781288147, 0.27575963735580444), (0.022922636941075325, 0.9651486277580261, -0.0317380428314209, 0.2587573826313019), (0.006070623193998153, -0.770042418911319, 0.029815238501249664)], 
    'right_knee': [(-0.1985805183649063, 0.4008098542690277, 0.29658448696136475), (0.031631484627723694, 0.9649024605751038, -0.02940165437757969, 0.2590332627296448), (-0.02741423975997844,)], 
    'right_ankle': [(-0.22984318435192108, -0.007531334646046162, 0.3131236433982849), (-0.03050919435918331, -0.2583571970462799, 0.10740445554256439, 0.9595754146575928), (0.030117010187545222, -0.01633065763028236, 0.005991738877577178)], 
    'right_shoulder': [(-0.18916499614715576, 1.3014750480651855, 0.37185773253440857), (-0.13724477589130402, -0.5742774605751038, 0.6120885610580444, 0.5260387063026428), (-0.02532097049637, 0.016461337593482624, -0.0256478949610282)], 
    'right_elbow': [(-0.0555269755423069, 1.2429389953613281, 0.6047157049179077), (-0.14571397006511688, -0.5721873044967651, 0.6197932958602905, 0.5169385075569153), (-0.00014755009593311967,)], 
    'right_wrist': [(0.06722392141819, 1.1939339637756348, 0.8273910284042358), (-0.14571397006511688, -0.5721873044967651, 0.6197932958602905, 0.5169385075569153), ()], 
    'left_hip': [(-0.16095951199531555, 0.8206321597099304, 0.10641399770975113), (0.014083372429013252, 0.982201874256134, 0.04612189158797264, -0.1815319061279297), (0.01792547001656103, -0.6528803228917127, -0.0028852713480175853)], 
    'left_knee': [(-0.17968063056468964, 0.4010468125343323, 0.07037650793790817), (0.02950858883559704, 0.9818595051765442, 0.0432649701833725, -0.18223392963409424), (0.024397620336189495,)], 
    'left_ankle': [(-0.2098942995071411, -0.006574932485818863, 0.039961978793144226), (0.018546290695667267, 0.183109313249588, 0.11222050338983536, 0.9764904379844666), (-0.013593623843457598, 0.07313760375509844, 0.06271052589314843)], 
    'left_shoulder': [(-0.16318313777446747, 1.3005445003509521, 0.006561741232872009), (0.5312401652336121, 0.3178047835826874, 0.224496990442276, 0.7525855898857117), (0.09327069960064104, -3.7050682331104943, -0.2749047122806273)], 
    'left_elbow': [(-0.16311581432819366, 1.208553671836853, -0.2523709535598755), (0.5405553579330444, 0.30168861150741577, 0.24702313542366028, 0.7454954981803894), (0.5601070748706386,)], 
    'left_wrist': [(-0.1522010862827301, 1.1325374841690063, -0.4996682405471802), (0.5405553579330444, 0.30168861150741577, 0.24702313542366028, 0.7454954981803894), ()]
}



C1_INDICES = [8, 14, 16]
C2_INDICES = [9, 10, 11, 12, 13, 14, 15]
C3_INDICES = [0, 1, 2, 4, 5, 6, 7, 14, 15, ]
C4_INDICES = [3, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]


class MotionSynthesizer():
    def __init__(self):
        # Initializer with no files
        self.c1 = [1.0]
        self.c2 = [1.0]
        self.c3 = [1.0]
        self.c4 = [1.0]

        self.current_reference_features = []
        self.current_features = []

        self._mocap = []
        self.generated_mocap = []

        self._lma = []

        self._reference = []

        self._extraction_framerate  = -1.0

        # TODO: CHANGE THIS TO GET THE ACTUAL CURRENT FRAMERATE!!!!
        self._current_framerate = -1.0

        self._models = {}
        for filename in os.listdir("../motion_synthesizer/models/"):
            f = os.path.join("../motion_synthesizer/models/", filename)
            if os.path.isfile(f):
                print(filename)
                model = xgb.XGBRegressor(verbosity=0)
                model.load_model(f)
                self._models[filename.split(".")[0]] = model

        self._frame_worth = -1.0 # How much each keyframe is worth (i.e if we have 1000 total frames and 500 keyframes then each keyframe is worth 2)


    def reset(self):
        self.c1 = [1.0]
        self.c2 = [1.0]
        self.c3 = [1.0]
        self.c4 = [1.0]

        self.current_reference_features = []
        self.current_features = []

        self._mocap = []
        self.generated_mocap = []

        self._lma = []

        #self._reference = []

        self._extraction_framerate  = -1.0

        # TODO: CHANGE THIS TO GET THE ACTUAL CURRENT FRAMERATE!!!!
        self._current_framerate = -1.0

    def set_current_mocap(self, mocap):
        self._mocap = []
        self.generated_mocap = []
        
        i = 0
        first = True
        for frame in mocap:
            frame = frame.copy()
            frame.pop("frame")
            self._mocap.append(frame)

            if(first):
                first = False
                i += 1
                continue
            
            if(i % self._extraction_framerate  == 0):
                gen = {"root": frame["root"][0], "neck": frame["neck"][0], "left_wrist":frame["left_wrist"][0], "right_wrist":frame["right_wrist"][0]}
                self.generated_mocap.append({'index': i, 'mocap': gen})

            i += 1

    def set_current_lma(self, lma):
        self._lma = []

        for frame in lma:
            frame = frame.copy()
            features = []

            for feature in frame["lma_features"]:
                if(type(feature) is tuple):
                    for i in range(len(feature)):
                        features.append(feature[i])
                else:
                    features.append(feature)

            frame["lma_features"] = features
            self._lma.append(frame)

        self._extraction_framerate = self._lma[0]["frame_counter"]

    def set_reference_lma(self, lma=None):
        if(lma == None):
            self._reference = ANGRY_LMA
        else:
            self._reference = lma
    
    def set_desired_pad(self, pad):
        lma = []
        pad_order = ["max_hand_distance",
          "avg_l_hand_hip_distance",
          "avg_r_hand_hip_distance",
          "max_stride_length",
          "avg_l_hand_chest_distance",
          "avg_r_hand_chest_distance",
          "avg_l_elbow_hip_distance",
          "avg_r_elbow_hip_distance",
          "avg_chest_pelvis_distance",
          "avg_neck_chest_distance",
          "avg_neck_rotation_w", "avg_neck_rotation_x", "avg_neck_rotation_y", "avg_neck_rotation_z",
          "avg_total_body_volume",
          "avg_triangle_area_hands_neck",
          "avg_triangle_area_feet_hips",
          
          "l_hand_speed",
          "r_hand_speed",
          "l_foot_speed",
          "r_foot_speed",
          "neck_speed",
          
          "l_hand_acceleration_magnitude",
          "r_hand_acceleration_magnitude",
          "l_foot_acceleration_magnitude",
          "r_foot_acceleration_magnitude",
          "neck_acceleration_magnitude",
        ]

        pad = np.asarray([pad])

        for feature in pad_order:
            lma.append(self._models[feature].predict(pad)[0])
        
        self._reference = lma
        print(self._reference)


    def compute_coefficients(self):
        self.compute_coefficient(1)
        self.compute_coefficient(2)
        self.compute_coefficient(3)
        self.compute_coefficient(4)

    def get_motion_changes(self):
        self.rule_1()
        self.rule_2()
        self.rule_3()
        self.rule_4()

        return self.generated_mocap

    def convert_single_frame(self, frame, counter):
        # Get the coefficients of the closest key frame and compute the changes of this single frame alone
        closest_index = int((counter/self._frame_worth)/self._extraction_framerate)
        if(closest_index >= len(self.c1)):
            closest_index = len(self.c1)-1

        frame_c1 = self.c1[closest_index]
        frame_c2 = self.c2[closest_index]
        frame_c3 = self.c3[closest_index]
        frame_c4 = self.c4[closest_index]

        generated = {"mocap": {"root": [], "neck": [], "left_wrist": [], "right_wrist": []}}
        root = self.rule_1_single(frame, frame_c1)
        neck = self.rule_2_single(frame, frame_c2)
        left_wrist, right_wrist = self.rule_3_single(frame, frame_c3)

        generated["mocap"]["root"] = root
        generated["mocap"]["neck"] = neck
        generated["mocap"]["left_wrist"] = left_wrist
        generated["mocap"]["right_wrist"] = right_wrist

        return generated

    def compute_coefficient(self, coefficient_number):
        feature_index = []
        if(coefficient_number == 1):
            print("== COMPUTING COEFFICIENT C1 - PELVIS ==")
            feature_index = C1_INDICES
        elif(coefficient_number == 2):
            print("== COMPUTING COEFFICIENT C2 - HEAD ==")
            feature_index = C2_INDICES
        elif(coefficient_number == 3):
            print("== COMPUTING COEFFICIENT C3 - HANDS ==")
            feature_index = C3_INDICES
        elif(coefficient_number == 4):
            print("== COMPUTING COEFFICIENT C4 - FRAMERATE ==")
            feature_index = C4_INDICES
        else:
            print("UNKNOWN COEFFICIENT!")

        self.reference_features = []
        for feature in feature_index:
            self.reference_features.append(self._reference[feature])

        self.reference_features = np.asarray(self.reference_features)

        self.current_features = []
        for features in [frame["lma_features"] for frame in self._lma]:
            temp = []
            for feature in feature_index:
                temp.append(features[feature])
            self.current_features.append(temp)

        self.current_features = np.asarray(self.current_features)

        coefficient = [1.0] * len(self.current_features)

        res = minimize(self.compute_sse,
                       coefficient,
                       method='Powell',
                       #callback = self.minimize_callback,
                       options={'maxiter': 50000, 'disp': False})

        coefficient = res.x
        print(str(coefficient) + "\n")

        if(coefficient_number == 1):
            self.c1 = coefficient
        elif(coefficient_number == 2):
            self.c2 = coefficient
        elif(coefficient_number == 3):
            self.c3 = coefficient
        elif(coefficient_number == 4):
            self.c4 = coefficient

    def compute_norms(self, reference, current_features, coefficient):
        norms = []

        for i in range(len(current_features)):
            norms.append(np.linalg.norm(
                reference - (current_features[i] * coefficient[i])) ** 2)

        return norms

    def compute_sse(self, coefficient):
        norms = self.compute_norms(
            self.reference_features, self.current_features, coefficient)
        return np.sum(norms)

    def minimize_callback(self, coefficient):
        print("Coefficient: " + str(coefficient) +
              " | SSE: " + str(self.compute_sse(coefficient)))

    def rule_1(self):
        print("\n== RULE 1 - PELVIS ==")
        t_pose_root_height = T_POSE["root"][0][1]
        first = True
        coefficient_index = 0
        for i in range(len(self._mocap)):
            if(first):
                # Ignore the first frame
                first = False
                continue

            if(i % self._extraction_framerate == 0):
                coefficient = self.c1[coefficient_index]
                coefficient_index += 1

                if(coefficient > 1.0):
                    t_pose_root_height = T_POSE["root"][0][1]

                    current_root_height = self._mocap[i]["root"][0][1]
                    new_root_height = current_root_height

                    if(current_root_height > t_pose_root_height):
                        new_root_height += (-current_root_height - -
                                            t_pose_root_height) * coefficient
                    else:
                        new_root_height += (-t_pose_root_height - -
                                            current_root_height) * (1.0 - (1.0/coefficient))

                    gen_index = next(index for index in range(
                        len(self.generated_mocap)) if self.generated_mocap[index]["index"] == i)
                        
                    print("Original:" + str(self.generated_mocap[gen_index]['mocap']['root']))

                    temp = self.generated_mocap[gen_index]['mocap']['root']
                    self.generated_mocap[gen_index]['mocap']['root'] = (
                        temp[0], new_root_height, temp[2])

                    print("Synthesized:" + str(self.generated_mocap[gen_index]['mocap']['root']))
                    print()

                else:
                    current_root_height = self._mocap[i]["root"][0][1]
                    new_root_height = current_root_height
                    new_root_height += (-current_root_height *
                                        (coefficient - 1.0))

                    gen_index = next(index for index in range(
                        len(self.generated_mocap)) if self.generated_mocap[index]["index"] == i)

                    print("Original:" + str(self.generated_mocap[gen_index]['mocap']['root']))

                    temp = self.generated_mocap[gen_index]['mocap']['root']
                    self.generated_mocap[gen_index]['mocap']['root'] = (
                        temp[0], new_root_height, temp[2])

                    print("Synthesized:" + str(self.generated_mocap[gen_index]['mocap']['root']))
                    print()

    def rule_2(self):
        print("\n== RULE 2 - HEAD ==")
        t_pose_root = T_POSE["root"][0]
        t_pose_head = T_POSE["neck"][0]

        dt_head = (t_pose_root[0] - t_pose_head[0], t_pose_root[1] -
                   t_pose_head[1], t_pose_root[2] - t_pose_head[2])

        first = True
        coefficient_index = 0
        for i in range(len(self._mocap)):
            if(first):
                # Ignore the first frame
                first = False
                continue

            if(i % self._extraction_framerate == 0):
                coefficient = self.c2[coefficient_index]
                coefficient_index += 1

                current_root_position = self._mocap[i]["root"][0]
                current_neck_position = self._mocap[i]["neck"][0]

                d_head = (current_root_position[0] - current_neck_position[0], current_root_position[1] -
                          current_neck_position[1], current_root_position[2] - current_neck_position[2])

                new_neck_position_x = current_neck_position[0]
                new_neck_position_y = current_neck_position[1]
                new_neck_position_z = current_neck_position[2]

                new_neck_position_x += dt_head[0] - \
                    ((dt_head[0]-d_head[0])/coefficient) - d_head[0]
                new_neck_position_y += dt_head[1] - \
                    ((dt_head[1]-d_head[1])/coefficient) - d_head[1]
                new_neck_position_z += dt_head[2] - \
                    ((dt_head[2]-d_head[2])/coefficient) - d_head[2]

                gen_index = next(index for index in range(
                    len(self.generated_mocap)) if self.generated_mocap[index]["index"] == i)

                print(self.generated_mocap[gen_index]['mocap']['neck'])
                self.generated_mocap[gen_index]['mocap']['neck'] = (
                    new_neck_position_x, new_neck_position_y, new_neck_position_z)
                print(self.generated_mocap[gen_index]['mocap']['neck'])
                print()

    def rule_3(self):
        print("\n== RULE 3 - HANDS ==")
        t_pose_root = T_POSE["root"][0]
        t_pose_chest = T_POSE["chest"][0]
        t_pose_head = T_POSE["neck"][0]
        t_pose_ground = (0.0, 0.0, 0.0)

        t_pose_left_hand = T_POSE["left_wrist"][0]
        t_pose_right_hand = T_POSE["right_wrist"][0]

        first = True
        coefficient_index = 0
        for i in range(len(self._mocap)):
            if(first):
                # Ignore the first frame
                first = False
                continue

            if(i % self._extraction_framerate == 0):
                coefficient = self.c2[coefficient_index]
                coefficient_index += 1

                current_root = self._mocap[i]["root"][0]
                current_chest = self._mocap[i]["chest"][0]
                current_head = self._mocap[i]["neck"][0]

                current_left_hand = self._mocap[i]["left_wrist"][0]
                current_right_hand = self._mocap[i]["right_wrist"][0]
                current_ground_left = (
                    current_left_hand[0], 0.0, current_left_hand[2])
                current_ground_right = (
                    current_right_hand[0], 0.0, current_right_hand[2])

                dl_head = (current_head[0] - current_left_hand[0], current_head[1] -
                           current_left_hand[1], current_head[2] - current_left_hand[2])
                dr_head = (current_head[0] - current_right_hand[0], current_head[1] -
                           current_right_hand[1], current_head[2] - current_right_hand[2])

                dl_chest = (current_chest[0] - current_left_hand[0], current_chest[1] -
                            current_left_hand[1], current_chest[2] - current_left_hand[2])
                dr_chest = (current_chest[0] - current_right_hand[0], current_chest[1] -
                            current_right_hand[1], current_chest[2] - current_right_hand[2])

                dl_root = (current_root[0] - current_left_hand[0], current_root[1] -
                           current_left_hand[1], current_root[2] - current_left_hand[2])
                dr_root = (current_root[0] - current_right_hand[0], current_root[1] -
                           current_right_hand[1], current_root[2] - current_right_hand[2])

                dl_ground = (current_ground_left[0] - current_left_hand[0], current_ground_left[1] -
                             current_left_hand[1], current_ground_left[2] - current_left_hand[2])
                dr_ground = (current_ground_right[0] - current_right_hand[0], current_ground_right[1] -
                             current_right_hand[1], current_ground_right[2] - current_right_hand[2])

                new_left_hand_position_x = current_left_hand[0]
                new_left_hand_position_y = current_left_hand[1]
                new_left_hand_position_z = current_left_hand[2]

                new_right_hand_position_x = current_right_hand[0]
                new_right_hand_position_y = current_right_hand[1]
                new_right_hand_position_z = current_right_hand[2]

                new_left_hand_position_x += (dl_head[0] * (coefficient - 1.0) + dl_chest[0] * (
                    coefficient - 1.0) + dl_root[0] * (coefficient - 1.0) + dl_ground[0] * (coefficient - 1.0))/4.0
                new_left_hand_position_y += (dl_head[1] * (coefficient - 1.0) + dl_chest[1] * (
                    coefficient - 1.0) + dl_root[1] * (coefficient - 1.0) + dl_ground[1] * (coefficient - 1.0))/4.0
                new_left_hand_position_z += (dl_head[2] * (coefficient - 1.0) + dl_chest[2] * (
                    coefficient - 1.0) + dl_root[2] * (coefficient - 1.0) + dl_ground[2] * (coefficient - 1.0))/4.0

                new_right_hand_position_x += (dr_head[0] * (coefficient - 1.0) + dr_chest[0] * (
                    coefficient - 1.0) + dr_root[0] * (coefficient - 1.0) + dr_ground[0] * (coefficient - 1.0))/4.0
                new_right_hand_position_y += (dr_head[1] * (coefficient - 1.0) + dr_chest[1] * (
                    coefficient - 1.0) + dr_root[1] * (coefficient - 1.0) + dr_ground[1] * (coefficient - 1.0))/4.0
                new_right_hand_position_z += (dr_head[2] * (coefficient - 1.0) + dr_chest[2] * (
                    coefficient - 1.0) + dr_root[2] * (coefficient - 1.0) + dr_ground[2] * (coefficient - 1.0))/4.0

                gen_index = next(index for index in range(
                    len(self.generated_mocap)) if self.generated_mocap[index]["index"] == i)

                print(self.generated_mocap[gen_index]
                      ['mocap']['left_wrist'])
                self.generated_mocap[gen_index]['mocap']['left_wrist'] = (
                    new_left_hand_position_x, new_left_hand_position_y, new_left_hand_position_z)
                self.generated_mocap[gen_index]['mocap']['right_wrist'] = (
                    new_right_hand_position_x, new_right_hand_position_y, new_right_hand_position_z)
                print(self.generated_mocap[gen_index]
                      ['mocap']['left_wrist'])
                print()

    def rule_4(self):
        print("\n== RULE 4 - FRAMERATE ==")
        print(self._current_framerate)
        self._current_framerate *= np.sum(self.c4)/len(self.c4)
        print(self._current_framerate)

    def rule_1_single(self, frame, coefficient):
        print("\n== RULE 1 SINGLE - PELVIS ==")
        t_pose_root_height = T_POSE["root"][0][1]
        generated_pose = ()
        
        if(coefficient > 1.0):
            t_pose_root_height = T_POSE["root"][0][1]
            current_root_height = frame["root"][1]

            new_root_height = current_root_height
            
            if(current_root_height > t_pose_root_height):
                new_root_height += (-current_root_height - -
                                    t_pose_root_height) * coefficient
            else:
                new_root_height += (-t_pose_root_height - -
                                    current_root_height) * (1.0 - (1.0/coefficient))
                
            generated_pose = (
                frame["root"][0], new_root_height, frame["root"][2])

        else:
            current_root_height = frame["root"][1]

            new_root_height = current_root_height
            new_root_height += (-current_root_height *
                                (coefficient - 1.0))

            generated_pose = (
                frame["root"][0], new_root_height, frame["root"][2])

        return generated_pose

    def rule_2_single(self, frame, coefficient):
        print("\n== RULE 2 SINGLE - HEAD ==")
        t_pose_root = T_POSE["root"][0]
        t_pose_head = T_POSE["neck"][0]

        dt_head = (t_pose_root[0] - t_pose_head[0], t_pose_root[1] -
                   t_pose_head[1], t_pose_root[2] - t_pose_head[2])

        current_root_position = frame["root"]
        current_neck_position = frame["neck"]

        d_head = (current_root_position[0] - current_neck_position[0], current_root_position[1] -
                  current_neck_position[1], current_root_position[2] - current_neck_position[2])

        new_neck_position_x = current_neck_position[0]
        new_neck_position_y = current_neck_position[1]
        new_neck_position_z = current_neck_position[2]

        new_neck_position_x += dt_head[0] - \
            ((dt_head[0]-d_head[0])/coefficient) - d_head[0]
        new_neck_position_y += dt_head[1] - \
            ((dt_head[1]-d_head[1])/coefficient) - d_head[1]
        new_neck_position_z += dt_head[2] - \
            ((dt_head[2]-d_head[2])/coefficient) - d_head[2]

        generated_pose = (
            new_neck_position_x, new_neck_position_y, new_neck_position_z)

        return generated_pose

    def rule_3_single(self, frame, coefficient):
        print("\n== RULE 3 SINGLE - HANDS ==")
        t_pose_root = T_POSE["root"][0]
        t_pose_chest = T_POSE["chest"][0]
        t_pose_head = T_POSE["neck"][0]
        t_pose_ground = (0.0, 0.0, 0.0)

        t_pose_left_hand = T_POSE["left_wrist"][0]
        t_pose_right_hand = T_POSE["right_wrist"][0]


        current_root = frame["root"]
        current_chest = frame["chest"]
        current_head = frame["neck"]
        current_left_hand = frame["left_wrist"]
        current_right_hand = frame["right_wrist"]

        current_ground_left = (
            current_left_hand[0], 0.0, current_left_hand[2])
        current_ground_right = (
            current_right_hand[0], 0.0, current_right_hand[2])

        dl_head = (current_head[0] - current_left_hand[0], current_head[1] -
                   current_left_hand[1], current_head[2] - current_left_hand[2])
        dr_head = (current_head[0] - current_right_hand[0], current_head[1] -
                   current_right_hand[1], current_head[2] - current_right_hand[2])

        dl_chest = (current_chest[0] - current_left_hand[0], current_chest[1] -
                    current_left_hand[1], current_chest[2] - current_left_hand[2])
        dr_chest = (current_chest[0] - current_right_hand[0], current_chest[1] -
                    current_right_hand[1], current_chest[2] - current_right_hand[2])

        dl_root = (current_root[0] - current_left_hand[0], current_root[1] -
                   current_left_hand[1], current_root[2] - current_left_hand[2])
        dr_root = (current_root[0] - current_right_hand[0], current_root[1] -
                   current_right_hand[1], current_root[2] - current_right_hand[2])

        dl_ground = (current_ground_left[0] - current_left_hand[0], current_ground_left[1] -
                     current_left_hand[1], current_ground_left[2] - current_left_hand[2])
        dr_ground = (current_ground_right[0] - current_right_hand[0], current_ground_right[1] -
                     current_right_hand[1], current_ground_right[2] - current_right_hand[2])

        new_left_hand_position_x = current_left_hand[0]
        new_left_hand_position_y = current_left_hand[1]
        new_left_hand_position_z = current_left_hand[2]
        new_right_hand_position_x = current_right_hand[0]
        new_right_hand_position_y = current_right_hand[1]
        new_right_hand_position_z = current_right_hand[2]

        new_left_hand_position_x += (dl_head[0] * (coefficient - 1.0) + dl_chest[0] * (
            coefficient - 1.0) + dl_root[0] * (coefficient - 1.0) + dl_ground[0] * (coefficient - 1.0))/4.0
        new_left_hand_position_y += (dl_head[1] * (coefficient - 1.0) + dl_chest[1] * (
            coefficient - 1.0) + dl_root[1] * (coefficient - 1.0) + dl_ground[1] * (coefficient - 1.0))/4.0
        new_left_hand_position_z += (dl_head[2] * (coefficient - 1.0) + dl_chest[2] * (
            coefficient - 1.0) + dl_root[2] * (coefficient - 1.0) + dl_ground[2] * (coefficient - 1.0))/4.0

        new_right_hand_position_x += (dr_head[0] * (coefficient - 1.0) + dr_chest[0] * (
            coefficient - 1.0) + dr_root[0] * (coefficient - 1.0) + dr_ground[0] * (coefficient - 1.0))/4.0
        new_right_hand_position_y += (dr_head[1] * (coefficient - 1.0) + dr_chest[1] * (
            coefficient - 1.0) + dr_root[1] * (coefficient - 1.0) + dr_ground[1] * (coefficient - 1.0))/4.0
        new_right_hand_position_z += (dr_head[2] * (coefficient - 1.0) + dr_chest[2] * (
            coefficient - 1.0) + dr_root[2] * (coefficient - 1.0) + dr_ground[2] * (coefficient - 1.0))/4.0


        l_generated_pose = (
            new_left_hand_position_x, new_left_hand_position_y, new_left_hand_position_z)
        r_generated_pose = (
            new_right_hand_position_x, new_right_hand_position_y, new_right_hand_position_z)
        
        return l_generated_pose, r_generated_pose


    """
    #OLD: USES A SINGLE COEFFICIENT
    def rule_1(self):
        if(self.c1 > 1.0):
            t_pose_root_height = T_POSE["root"][0][1]
            first = True
            for i in range(len(self._mocap)):
                if(first):
                    # Ignore the first frame
                    first = False
                    continue

                if(i % self._extraction_framerate == 0):
                    current_root_height = self._mocap[i]["root"][0][1]
                    new_root_height = current_root_height

                    if(current_root_height > t_pose_root_height):
                        new_root_height += (current_root_height - t_pose_root_height) * self.c1
                    else:
                        new_root_height += (t_pose_root_height - current_root_height) * (1.0 - (1.0/self.c1))  


                    gen_index = next(index for index in range(len(self.generated_mocap)) if self.generated_mocap[index]["index"] == i)
                    print(self.generated_mocap[gen_index]['mocap']['root'][0])
                    temp = self.generated_mocap[gen_index]['mocap']['root'][0]
                    self.generated_mocap[gen_index]['mocap']['root'][0] = (temp[0], new_root_height, temp[2])

                    print(self.generated_mocap[gen_index]['mocap']['root'][0])
                    print()
               
        else:
            first = True
            for i in range(len(self._mocap)):
                if(first):
                    # Ignore the first frame
                    first = False
                    continue

                if(i % self._extraction_framerate == 0):
                    current_root_height = self._mocap[i]["root"][0][1]
                    new_root_height = current_root_height
                    new_root_height += (current_root_height * (self.c1 - 1.0))

                    gen_index = next(index for index in range(len(self.generated_mocap)) if self.generated_mocap[index]["index"] == i)

                    print(self.generated_mocap[gen_index]['mocap']['root'][0])
                    temp = self.generated_mocap[gen_index]['mocap']['root'][0]
                    self.generated_mocap[gen_index]['mocap']['root'][0] = (temp[0], new_root_height, temp[2])
                    print(self.generated_mocap[gen_index]['mocap']['root'][0])
                    print()
    """

"""
ms = MotionSynthesizer("neutral_walk_mocap.txt",
                       "neutral_walk_lma.txt", ANGRY_LMA)
ms.compute_coefficients()
ms.rule_1()
ms.rule_2()
ms.rule_3()
ms.rule_4()
"""

# Coefficients
# c1 - Pelvis
# c2 - Head
# c3 - Hands
# c4 - FrameRate
# c5(?) - Elbows
# c6(?) - Head Rotation

# Position Features
#                     [
# 0                      c3 - max hand_distance (1D),
# 1                      c3 - average l_hand_hip_distance (1D),
# 2                      c3 - average r_hand_hip_distance (1D),
# 3                      c4(?) - max stride length (distance between left and right foot) (1D),
# 4                      c3 - average l_hand_chest_distance (1D),
# 5                      c3 - average r_hand_chest_distance (1D),
# 6                      c3(?) - average l_elbow_hip_distance (1D),
# 7                      c3(?) - average r_elbow_hip_distance (1D),
# 8                      c1 - average chest_pelvis_distance (1D),
# 9                      c2 - average neck_chest_distance (1D),
#
# 10,11,12,13            c2 - average neck_rotation (4D)
#
# 14                     c1, c2, c3 - average total_body_volume (1D)
#
# 15                     c2, c3 - triangle area between hands and neck (1D)
# 16                     c1 - triangle area between feet and root (1D)
#
#
# Movement Features:
#
# 17                     c4 - l_hand speed (1D)
# 18                     c4 - r_hand speed (1D)
# 19                     c4 - l_foot_speed (1D)
# 20                     c4 - r_foot_speed (1D)
# 21                     c4 - neck speed (1D)
#
# 22                     c4 - l_hand acceleration magnitude (1D)
# 23                     c4 - r_hand acceleration magnitude (1D)
# 24                     c4 - l_foot acceleration magnitude (1D)
# 25                     c4 - r_foot acceleration magnitude (1D)
# 26                     c4 - neck acceleration magnitude (1D)
#                     ]


# 1 - GET MOCAP FILE
# 2 - GET MOCAP LMA FILE
# 3 - GET REFERENCE LMA
#4 - SYNTHESIZE
# 5a - STANDERDIZE NEW LMA FEATURES
# 5b - EMOTION CLASSIFICATION TO SEE EMOTION OF NEW FEATURES

# Note that we apply motion synthesis only on sampled keyframes, i.e., every 5fth frame uniformly drawn from the motion
# We then interpolate the edited keyframes to obtain the full motion (NOTE: Pelo menos para o mocap do neutral walk aquilo esta de 5 em 5 keyframes)

# we 1st optimize for poses that satisfy the desired features as best as possible,
# To this end, we designed four heuristic rules as shown in Table 2 to modify the positions of four control joints: head, hands, and pelvis

# Portanto temos 4 regras heuristicas, 1 para cada core joint (cabeca, l_hand r_hand e pelvis). Cada regra diz respeito a um set de LMA features diferente
# Cada uma destas regras depois usa um coeficiente para calcular a nova posicao destas core joints

# We search for c1, the coefficient for rule g1, that minimizes the distance between the current feature values and the desired feature values for all the
# keyframes indexed by time t in a least-squares sense (ou seja, encontramos o coeficiente que minimiza para TODAS as keyframes)
# To search for each coefficient we use interior-relective Newton method

#  From these optimized coeffcients we then compute the desired positions for all the control joints on each keyframe.
#  We then pass these joint positions to an Inverse Kinematics solver (NOTE: Not sure se obrigatorio, try first without it, mas e capaz de ser preciso para encontrarmos a posiçao de cenas tipo elbows e shoulders given the hands)
