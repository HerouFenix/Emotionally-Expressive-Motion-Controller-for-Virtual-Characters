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
    'frame': [-0.16609051983616052, 0.8208358718372436, 0.1907889500265707, 0.9992657024895122, 2.3401096336136264e-05, -0.03506671168169595, -0.015439592363896653, 1.0, 0.0, 0.0, -0.0, 0.998739717777699, -0.001630854403921047, -6.348387706985323e-06, 0.05016289870944029, 0.2702945424630904, 0.005101763279313845, 0.9623726356646698, -0.02745437032508739, 0.015685323692000902, 0.0040567961657829886, 0.1323987319214238, 0.990590709904821, -0.0344116788420891, 0.5379828762356462, -0.10424622637951891, -0.5593772633964808, 0.62193586997207, 0.029127925921603046, 0.1791379887572908, 0.00041021193944449903, -0.9828828670486963, -0.0430195580206552, 0.03538608537932216, 0.00944877943667258, -0.13356901861722095, -0.9896879250324737, -0.050870904038617604, 0.732558432647076, 0.5271783355415982, 0.36408710423521307, 0.22996027009652464, 0.05363700522404034],
    'root': [(-0.16393067900481517, 0.8908024984163395, 0.19086802197713065), (2.3401096336136264e-05, -0.03506671168169595, -0.015439592363896653, 0.9992657024895122), [(0.005905173415750418, -0.00016419633563590966, -0.003921920256091087), (0.03025022027045334, 0.0332027621317227, 0.03045228333064668)]],
    'chest': [(-0.15510152738009653, 1.176817072162552, 0.19119125794462608), (2.3401096336136264e-05, -0.03506671168169595, -0.015439592363896653, 0.9992657024895122), (-6.543537966809611e-120, 8.179422673323876e-121, -6.543538052734349e-120)],
    'neck': [(-0.16396693174523297, 1.455237387835983, 0.18970764591934747), (-0.003365431189922262, -0.035004855594094435, 0.03464874124794187, 0.9987806559438742), (-0.03699513046845043, 7.010309914870117e-05, -0.01995973880519428)],
    'right_hip': [(-0.18440978862585763, 0.6115219943066721, 0.2853593207795824), (0.02092571794128974, 0.9521095003614457, -0.03140602556637932, 0.3034193059253874), (-0.0029210802978426012, -0.7540526791453281, -0.0022187088694792705)],
    'right_knee': [(-0.21120727386348392, 0.20123800716411672, 0.3029276357359328), (0.028392070707210865, 0.951916108000509, -0.02902546910067863, 0.30365627902878184), (-0.01968545596804811,)],
    'right_ankle': [(-0.18638028500625672, -0.019624550371823063, 0.3392795451641958), (-0.036314215775609164, -0.30179490214374505, 0.10847463688140054, 0.9464852708447702), (-0.002075642990414972, 0.00012379026127057278, -0.02084720355855131)],
    'right_shoulder': [(-0.11978389523184702, 1.2702817995221731, 0.4905300058243852), (-0.13460289200718573, -0.5762368392156241, 0.6095042852976471, 0.5275771913169611), (0.023274057840855338, -0.02833052709610294, -0.00236061524517822)],
    'right_elbow': [(0.003103773887990352, 1.2168324420838816, 0.7071979309937888), (-0.14298061232194498, -0.5742154455480566, 0.6171229888116267, 0.5186447563354842), (-0.0005355704293195668,)],
    'right_wrist': [(0.06923307696771866, 1.1893999243474414, 0.8262804278287369), (-0.14298061232194498, -0.5742154455480566, 0.6171229888116267, 0.5186447563354842), ()],
    'left_hip': [(-0.1684085525398451, 0.6117012522270966, 0.0879133640546363), (0.013252653623734631, 0.9884482456096182, 0.045762402142449275, -0.1438757640235142), (0.0070176764879616655, -0.13417069794967273, -0.02175175267079838)],
    'left_knee': [(-0.1913854046952981, 0.20224037086508273, 0.05428319892030012), (0.0307383239154728, 0.988059068335325, 0.04320977228917545, -0.144662878797274), (0.014768407376192198,)],
    'left_ankle': [(-0.16005811081679322, -0.018573621782912303, 0.023965793347661607), (0.012113650222451677, 0.14829925600007626, 0.10932014311163026, 0.9828070494548584), (-0.0012416624886192368, 0.006605340460438594, -0.021743639623578743)],
    'left_shoulder': [(-0.16123305355591694, 1.2535233949881845, -0.12478107583530965), (0.5243657784573047, 0.32998654070706845, 0.23697593799455025, 0.7483260106041018), (-0.0588906837057019, 0.20977411724927078, 0.016701846231030076)],
    'left_elbow': [(-0.15473282054814594, 1.1720310891608234, -0.36600121203087255), (0.5330259035191006, 0.315806860030936, 0.25695730020064617, 0.7417023386835464), (-0.025339971194352405,)],
    'left_wrist': [(-0.1485489583038482, 1.1303869136232683, -0.49841643509512956), (0.5330259035191006, 0.315806860030936, 0.25695730020064617, 0.7417023386835464), ()]
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
