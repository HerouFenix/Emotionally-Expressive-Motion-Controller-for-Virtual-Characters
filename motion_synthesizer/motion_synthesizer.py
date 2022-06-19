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

PRESET_EMOTIONS = {
    # NORMAL
    (0.05, -0.1, 0.0): [ 0.5186799,   0.46649932,  0.15772843,  0.64237505,  0.37424588,  0.18964156,
   0.12902201,  0.24415847,  0.23615103,  0.22389398,  0.33483888,  0.09003431,
   0.1015784 ,  0.07853313,  0.2298812 , -0.16199612, -0.15539091, -0.20247241,
  -0.12584846, -0.15541993,  0.02993792,  0.02793053,  0.03596406,  0.04410392,
   0.03035523],
    
    # TIRED
    (0.1, -0.7, -0.2): [ 0.61826428,  0.29773832,  0.31631444,  0.61099298,  0.38177025,  0.4067126,
   0.27889606,  0.27638706,  0.23615097,  0.22389406,  0.41153708,  0.08924712,
   0.10776476,  0.14451101,  0.22792978, -0.03813869, -0.07015731, -0.05813905,
  -0.04415843, -0.06872438,  0.00450561,  0.00533833,  0.023505  ,  0.01636196,
   0.00294026],
    
    # TIRED 2
    (0.1, -0.75, -0.25): [ 0.4450052,   0.30267617,  0.44533758,  0.29547794,  0.23564595,  0.35828984,
   0.11714197,  0.13302721,  0.236151  ,  0.22389399,  0.20719219,  0.07022425,
   0.06840402,  0.06758865,  0.08148723, -0.13730852, -0.13381774, -0.05919065,
  -0.14447553, -0.1255996 ,  0.01493876,  0.02008506,  0.00624596,  0.01632991,
   0.01687064],
    
    # EXHAUSTED
    (-0.1, -0.6, -0.15): [ 0.62813831,  0.33296396,  0.29231774,  0.23793207,  0.41810269,  0.37391618,
   0.27755375,  0.278451  ,  0.23615101,  0.22389397,  0.19359697,  0.06910677,
   0.05304403,  0.14552894,  0.08755464, -0.01530395, -0.03951186, -0.06149991,
  -0.00867908, -0.03103101,  0.00392392,  0.00452358,  0.00586803,  0.00200847,
   0.00397053],
    
    # ANGRY
    (-0.5, 0.8, 0.9): [ 0.4849567,   0.43509817,  0.40199345,  0.72839327,  0.33868578,  0.19559014,
   0.41532856,  0.26116835,  0.23615097,  0.22389401,  0.4839097 ,  0.10084644,
   0.0859726 ,  0.02596105,  0.21177863, -0.2379167 , -0.21990552, -0.16867742,
  -0.28235362, -0.21497578,  0.02257726,  0.07411671,  0.0837294 ,  0.01054426,
   0.02439681],
    
    # HAPPY
    (0.8, 0.5, 0.15): [ 0.62895403,  0.28924824,  0.33817309,  0.30717667,  0.39698883,  0.42866225,
   0.23510907,  0.26563808,  0.23615096,  0.22389403,  0.32104657,  0.02306457,
   0.0543723 ,  0.15397796,  0.09533507, -0.02423276, -0.15098949, -0.11906824,
  -0.03863518, -0.07934174,  0.04183279,  0.03219935,  0.0057589 ,  0.01831188,
   0.01759122],
    
    # HAPPY 2
    (0.6, 0.4, 0.1): [ 0.54932366,  0.14031404,  0.43870564,  0.3856451,   0.17669227,  0.42007307,
   0.09711947,  0.25737955,  0.23615102,  0.223894  ,  0.35843939,  0.08044427,
   0.05713033,  0.09086024,  0.1029739 , -0.15114259, -0.18987385, -0.17932219,
  -0.07464918, -0.1615106 ,  0.02929636,  0.02472049,  0.02621871,  0.01596762,
   0.01032188],
    
    # SAD
    (-0.6, -0.4, -0.3): [ 0.48203249,  0.30429993,  0.32106327,  0.3275376,   0.17515474,  0.30992711,
   0.25770942,  0.38230947,  0.23615101,  0.223894  ,  0.35171322,  0.08729186,
   0.05838501,  0.05864761,  0.09994176, -0.07459375, -0.13680961, -0.13222726,
  -0.05183191, -0.08656714,  0.0209568 ,  0.02331046,  0.00706071,  0.01917414,
   0.02123904],
    
    # PROUD
    (0.4, 0.2, 0.35): [ 0.86942627,  0.39191253,  0.48918249,  0.54010186,  0.44352916,  0.47570189,
   0.27260754,  0.33014613,  0.236151  ,  0.22389407,  0.48002736,  0.17669562,
   0.07204263,  0.16264799,  0.20788395, -0.0481359 , -0.05412584, -0.04150883,
  -0.03012454, -0.05175006,  0.01138745,  0.00472769,  0.01078721,  0.00637944,
   0.00644837],
    
    # CONFIDENT
    (0.3, 0.3, 0.9): [ 0.72487293,  0.35887094,  0.42138989,  0.78525169,  0.31323459,  0.41962872,
   0.3456292 ,  0.36508856,  0.236151  ,  0.22389399,  0.64049156,  0.14345473,
   0.15673073,  0.10815576,  0.27122958, -0.12987   , -0.15814968, -0.13852751,
  -0.14657472, -0.14309478,  0.02549621,  0.01590126,  0.01748402,  0.03745181,
   0.01475212],
    
    # CONFIDENT 2
    (0.25, 0.15, 0.4): [ 0.74932451,  0.3292823,   0.43148791,  0.66676494,  0.416184,    0.44834246,
   0.29014289,  0.29981556,  0.236151  ,  0.22389402,  0.59168066,  0.08334877,
   0.0842501 ,  0.16201821,  0.24783783, -0.09810907, -0.09096309, -0.06815321,
  -0.07576912, -0.09878263,  0.01330656,  0.00623767,  0.00886188,  0.02393857,
   0.00483165],
    
    # CONFIDENT 3
    (0.3, 0.4, 0.6): [ 0.83631549,  0.4374688,   0.41346408,  0.68161331,  0.43397967,  0.45622925,
   0.37222944,  0.33910005,  0.23615101,  0.22389399,  0.58342321,  0.14029728,
   0.10669285,  0.15740731,  0.24650132, -0.07381996, -0.07477095, -0.03838059,
  -0.11446716, -0.08053586,  0.0059353 ,  0.00862793,  0.00701826,  0.01004484,
   0.00694983],
    
    # AFRAID
    (-0.6, 0.7, -0.8): [ 2.08340828e-01,  2.64129062e-01,  2.27426258e-01,  2.30559942e-01,
   1.28220619e-01,  1.26505977e-01,  2.62098755e-01,  3.14412678e-01,
   2.36151019e-01,  2.23893942e-01,  2.06627529e-01,  5.68116833e-02,
   3.86782264e-02,  2.64402409e-02,  9.19677408e-02, -1.91390101e-02,
  -2.03380232e-02, -1.51719401e-02, -1.05658747e-02, -1.84225226e-02,
   9.78201230e-04,  7.71277062e-04,  1.90137609e-03,  1.32488543e-03,
   2.00411355e-04],
    
    # ACTIVE
    (0.1, 0.6, 0.4): [ 0.64517713,  0.45294615,  0.38687914,  0.25322463,  0.28559495,  0.40231506,
   0.24608379 , 0.39503441,  0.23615103 , 0.22389397 , 0.34467945,  0.12886602,
   0.06498919 , 0.06585845,  0.07055107 ,-0.08868541 ,-0.07737261, -0.0382299,
  -0.11474913 ,-0.07528836,  0.01771545 , 0.01259181 , 0.00699753,  0.01187604,
   0.00587643],
    
    # ANGRY 2
    (-0.5, 0.7, 0.9): [ 0.35083197,  0.55445003,  0.53397469,  0.16899797,  0.44933407,  0.42984391,
   0.34636631,  0.36784071 , 0.23615104 , 0.22389401 , 0.39055913 , 0.13535529,
   0.0112442 ,  0.07009147 , 0.07013562 ,-0.02260682 ,-0.03678275 ,-0.00227649,
  -0.00205315, -0.00272447 , 0.03883486 , 0.00842329 , 0.00230566 , 0.00163432,
   0.00361933],

    # HAPPY 3
    (0.6, 0.5, 0.2): [ 0.87840172,  0.40800089,  0.69014457,  0.60317439,  0.4222134,   0.57671535,
   0.22764189,  0.396286  ,  0.236151  ,  0.223894  ,  0.5703048 ,  0.1081096,
   0.14953151,  0.14243552,  0.22793017, -0.09510082, -0.15857715, -0.06167734,
  -0.07744886, -0.07451435,  0.05870626,  0.06389066,  0.03951913,  0.04674497,
   0.04283095],

    # SAD 2
    (-0.6, -0.3, -0.3): [ 6.07361422e-02,  4.73387724e-01,  4.24869513e-01,  3.19499137e-01,
   2.75486194e-01,  2.23666970e-01,  2.39500768e-01,  2.23175646e-01,
   2.36150994e-01,  2.23894031e-01,  1.72089341e-01,  6.15466172e-02,
   3.93021977e-02,  4.70122402e-03,  1.31860887e-01, -3.34407410e-02,
  -4.12166130e-02, -6.68352569e-04, -4.60027201e-04, -1.50126704e-03,
   1.14945239e-02,  1.41309644e-02,  2.74578227e-03,  3.15886054e-03,
   8.74024321e-04],

    # DISGUSTED
    (-0.4, 0.25, -0.1): [ 0.36427822,  0.14763415,  0.23171892,  0.3113933,   0.31870399,  0.36442134,
   0.16946413 , 0.22656426,  0.23615096,  0.22389405,  0.10435639,  0.01964235,
   0.04378811 , 0.09296647,  0.12856039, -0.00715507, -0.00746254, -0.00743695,
  -0.00758209 ,-0.00526928,  0.00302902,  0.00302307,  0.00269212,  0.00412398,
   0.00252165],

    # AFRAID 2
    (-0.5, 0.7, -0.8): [ 0.17382527,  0.47317039,  0.45626894,  0.32589129,  0.26317262,  0.25809266,
   0.223039  ,  0.23820646,  0.23615101,  0.22389399,  0.18209588,  0.03709195,
   0.06255196,  0.01292796,  0.1270879 , -0.02234098, -0.02249944, -0.01096152,
  -0.0250001 , -0.01976945,  0.0124769 ,  0.01528433,  0.00939255,  0.03051273,
   0.01267987],

    # NEUTRAL 2
    (0.0, 0.0, 0.0): [ 0.65396667,  0.39230998,  0.26207914,  0.28074901,  0.43823131,  0.37012581,
   0.29183602,  0.22392862,  0.23615098,  0.22389404,  0.29173512,  0.10834897,
   0.05081195,  0.14995357,  0.0809965 , -0.00239144, -0.01068324, -0.00267103,
  -0.01422459, -0.00402122,  0.00279384,  0.00551916,  0.00357097,  0.00630504,
   0.00334763],
}



# TODO: CHANGE T-POSE TO GET LOCAL FRAME INSTEAD OF CENTER OF MASS!!!!!
T_POSE = {
    'frame': [0., 0.88954026, 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.], 
    'root': [(0.0, 0.8895402550697327, 0.0)], 
    'chest': [(0.0, 1.125691294670105, 0.0)], 
    'neck': [(0.0, 1.3495852947235107, 0.0)], 
    'right_hip': [(0.0, 0.8895402550697327, 0.08488699793815613)], 
    'right_knee': [(0.0, 0.4679942727088928, 0.08488699793815613)], 
    'right_ankle': [(0.0, 0.05812425911426544, 0.08488699793815613)], 
    'right_shoulder': [(-0.024049999192357063, 1.369191288948059, 0.18310999870300293)], 
    'right_elbow': [(-0.024049999192357063, 1.0944032669067383, 0.18310999870300293)], 
    'right_wrist': [(-0.024049999192357063, 0.8354562520980835, 0.18310999870300293)], 
    'left_hip': [(0.0, 0.8895402550697327, -0.08488699793815613)], 
    'left_knee': [(0.0, 0.4679942727088928, -0.08488699793815613)], 
    'left_ankle': [(0.0, 0.05812425911426544, -0.08488699793815613)], 
    'left_shoulder': [(-0.024049999192357063, 1.369191288948059, -0.18310999870300293)], 
    'left_elbow': [(-0.024049999192357063, 1.0944032669067383, -0.18310999870300293)], 
    'left_wrist': [(-0.024049999192357063, 0.8354562520980835, -0.18310999870300293)]
}

#00          c3_hands           "max_hand_distance",
#01          c3l_hips           "avg_l_hand_hip_distance",
#02          c3r_hips           "avg_r_hand_hip_distance",
#03          --                 "max_stride_length",
#04          c3l_chest          "avg_l_hand_chest_distance",
#05          c3r_chest          "avg_r_hand_chest_distance",
#06          c4_l               "avg_l_elbow_hip_distance",
#07          c4_r               "avg_r_elbow_hip_distance",
#08          c1                 "avg_chest_pelvis_distance",
#09          c2                 "avg_neck_chest_distance",
#          
#10          c1, c2,c3_hands    "avg_total_body_volume",
#11          c1                 "avg_lower_body_volume",
#12          c2,c3_hands        "avg_upper_body_volume",
#          
#13          c3_hands, c3_chest "avg_triangle_area_hands_neck",
#14          c1                 "avg_triangle_area_feet_hips",

C1_INDICES = [8, 10, 11, 14]
C2_INDICES = [9, 10, 12]

C3_INDICES = [0,1,2,4,5,12,13]
C3_HANDS_INDICES = [0, 10, 12, 13]

C3_LEFT_HIPS_INDICES = [1]
C3_RIGHT_HIPS_INDICES = [2]

C3_LEFT_CHEST_INDICES = [4, 13]
C3_RIGHT_CHEST_INDICES = [5, 13]

C4_INDICES = [6,7]
C4_LEFT_INDICES = [6]
C4_RIGHT_INDICES = [7]


class MotionSynthesizer():
    def __init__(self):
        # Initializer with no files
        self.c1 = [1.0]
        self.c2 = [1.0]

        self.c3 = [1.0]
        self.c3_1 = [1.0] #C3_HANDS_INDICES
        self.c3_2 = [1.0] #C3_LEFT_HIPS_INDICES
        self.c3_3 = [1.0] #C3_RIGHT_HIPS_INDICES
        self.c3_4 = [1.0] #C3_LEFT_CHEST_INDICES
        self.c3_5 = [1.0] #C3_RIGHT_CHEST_INDICES

        self.c4 = [1.0]
        self.c4_1 = [1.0] #C4_LEFT
        self.c4_2 = [1.0] #C4_RIGHT

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
        """
        for filename in os.listdir("../motion_synthesizer/models/"):
            f = os.path.join("../motion_synthesizer/models/bandai_kin/5frame/xgb/", filename)
            if os.path.isfile(f):
                print(filename)
                model = xgb.XGBRegressor(verbosity=0)
                model.load_model(f)
                self._models[filename.split(".")[0]] = model
        """

        self._desired_emotion = np.asarray([0.0, 0.0, 0.0])


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
        
        i = 1
        for frame in mocap:
            frame = frame.copy()
            frame.pop("frame")
            self._mocap.append(frame) # This already ignores the first frame, which is why we start i at 1

            gen = {"root": frame["root"][0], "neck": frame["neck"][0], "left_wrist":frame["left_wrist"][0], "right_wrist":frame["right_wrist"][0], "left_elbow":frame["left_elbow"][0], "right_elbow":frame["right_elbow"][0]}
            orn = {"root": frame["root"][1], "neck": frame["neck"][1], "left_wrist":frame["left_wrist"][1], "right_wrist":frame["right_wrist"][1], "left_elbow":frame["left_elbow"][1], "right_elbow":frame["right_elbow"][1]}
            
            self.generated_mocap.append({'index': i, 'mocap': gen, 'orn': orn})

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
            #self._reference = PRESET_EMOTIONS[(0.3, 0.3, 0.9)] # Confident
            #self._desired_emotion = np.asarray([0.3, 0.3, 0.9])

            #self._reference = PRESET_EMOTIONS[(-0.5, 0.8, 0.9)] # Angry
            #self._desired_emotion = np.asarray([-0.5, 0.8, 0.9])

            self._reference = PRESET_EMOTIONS[(-0.6, 0.7, -0.8)] # Afraid
            self._desired_emotion = np.asarray([-0.6, 0.7, -0.8])

        else:
            self._reference = lma
    
    def set_desired_pad(self, pad):
        lma = []
        pad_order = [
            "max_hand_distance",
            "avg_l_hand_hip_distance",
            "avg_r_hand_hip_distance",
            "max_stride_length",
            "avg_l_hand_chest_distance",
            "avg_r_hand_chest_distance",
            "avg_l_elbow_hip_distance",
            "avg_r_elbow_hip_distance",
            "avg_chest_pelvis_distance",
            "avg_neck_chest_distance",
          
            "avg_total_body_volume",
            "avg_lower_body_volume",
            "avg_upper_body_volume",
          
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
        self._desired_emotion = pad

        for feature in pad_order:
            lma.append(self._models[feature].predict(pad)[0])
        
        self._reference = lma
        print(self._reference)


    def compute_coefficients(self):
        self.compute_coefficient(1)
        self.compute_coefficient(2)
        
        
        #self.compute_coefficient(3)
        self.compute_coefficient(3.1)
        self.compute_coefficient(3.2)
        self.compute_coefficient(3.3)
        self.compute_coefficient(3.4)
        self.compute_coefficient(3.5)
        
        
        #self.compute_coefficient(4)
        self.compute_coefficient(4)
        #self.compute_coefficient(4.1)
        #self.compute_coefficient(4.2)

    def get_motion_changes(self):
        self.rule_1()
        self.rule_2()
        self.rule_3()
        self.rule_4()

        return self.generated_mocap

    def convert_single_frame(self, frame, counter):
        # Get the coefficients of the closest key frame and compute the changes of this single frame alone
        #closest_index = int((counter/self._frame_worth)/self._extraction_framerate)
        closest_index = int(counter/5)

        if(closest_index >= len(self.c1)):
            closest_index = len(self.c1)-1

        frame_c1 = self.c1[closest_index]
        frame_c2 = self.c2[closest_index]
        frame_c3 = 0.0
        frame_c4 = 0.0

        generated = {"mocap": {"root": [], "neck": [], "left_wrist": [], "right_wrist":[], "left_elbow":[], "right_elbow":[]}}
        root = self.rule_1_single(frame, frame_c1)
        neck = self.rule_2_single(frame, frame_c2)
        left_wrist, right_wrist = self.rule_3_single(frame, frame_c3)
        left_elbow, right_elbow = self.rule_4_single(frame, frame_c4)

        generated["mocap"]["root"] = root
        generated["mocap"]["neck"] = neck
        generated["mocap"]["left_wrist"] = left_wrist
        generated["mocap"]["right_wrist"] = right_wrist
        generated["mocap"]["left_elbow"] = left_elbow
        generated["mocap"]["right_elbow"] = right_elbow

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
        elif(coefficient_number == 3.1):
            print("== COMPUTING COEFFICIENT C3 - HANDS-HANDS ==")
            feature_index = C3_HANDS_INDICES
        elif(coefficient_number == 3.2):
            print("== COMPUTING COEFFICIENT C3 - L HAND-HIPS ==")
            feature_index = C3_LEFT_HIPS_INDICES
        elif(coefficient_number == 3.3):
            print("== COMPUTING COEFFICIENT C3 - R HAND-HIPS ==")
            feature_index = C3_RIGHT_HIPS_INDICES
        elif(coefficient_number == 3.4):
            print("== COMPUTING COEFFICIENT C3 - L HAND-CHEST ==")
            feature_index = C3_LEFT_CHEST_INDICES
        elif(coefficient_number == 3.5):
            print("== COMPUTING COEFFICIENT C3 - R HAND-CBEST ==")
            feature_index = C3_RIGHT_CHEST_INDICES

        elif(coefficient_number == 4):
            print("== COMPUTING COEFFICIENT C4 - ELBOWS ==")
            feature_index = C4_INDICES
        elif(coefficient_number == 4.1):
            print("== COMPUTING COEFFICIENT C4 - LEFT ELBOW ==")
            feature_index = C4_LEFT_INDICES
        elif(coefficient_number == 4.2):
            print("== COMPUTING COEFFICIENT C4 - RIGHT ELBOW ==")
            feature_index = C4_RIGHT_INDICES

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
        elif(coefficient_number == 3.1):
            self.c3_1 = coefficient
        elif(coefficient_number == 3.2):
            self.c3_2 = coefficient
        elif(coefficient_number == 3.3):
            self.c3_3 = coefficient
        elif(coefficient_number == 3.4):
            self.c3_4 = coefficient
        elif(coefficient_number == 3.5):
            self.c3_5 = coefficient

            
        elif(coefficient_number == 4):
            self.c4 = coefficient
        elif(coefficient_number == 4.1):
            self.c4_1 = coefficient
        elif(coefficient_number == 4.2):
            self.c4_2 = coefficient

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
        # Move on Y axis

        t_pose_root_height = T_POSE["root"][0][1]

        # Vector from Pelvis to Ground in T-Pose
        #dt_pelvis = np.array(
        #            [0.0, 
        #             0.0 - t_pose_root_height,
        #             0.0]
        #            )

        coefficient_index = 0
        for i in range(len(self._mocap)):
            coefficient = self.c1[coefficient_index]
            coefficient_index += 1
            
            current_root_height = self._mocap[i]["root"][0][1]
            
            # Vector from Pelvis to Ground in Current Pose
            #d_pelvis = np.array(
            #            [0.0, 
            #             0.0 - current_root_height,
            #             0.0]
            #            )

            new_root_height = current_root_height

            # If Coefifcient > 1.0 -> Move Up (new root height += positive value) ; Else Move Down (new root height += negative value)
            #e.g coefficient = 1.2 -> new height += 0.2
            #e.g coefficient = 0.8 -> new height += -0.2
            new_root_height += 1.0 * ((coefficient - 1.0) * 0.075) #0.075 -> dampening factor

                
            gen_index = i
            print("Original:" + str(self.generated_mocap[gen_index]['mocap']['root']))

            temp = self.generated_mocap[gen_index]['mocap']['root']
            self.generated_mocap[gen_index]['mocap']['root'] = (
                temp[0], new_root_height, temp[2])

            print("Synthesized:" + str(self.generated_mocap[gen_index]['mocap']['root']))
            print()


    def rule_2(self):
        print("\n== RULE 2 - HEAD ==")
        # Move on the X and Y axis

        t_pose_root = T_POSE["root"][0]
        t_pose_head = T_POSE["neck"][0]
        
        # Vector from Head to Pelvis in T-Pose
        dt_head = (t_pose_root[0] - t_pose_head[0], 
                   t_pose_root[1] - t_pose_head[1],
                   t_pose_root[2] - t_pose_head[2])

        coefficient_index = 0
        for i in range(len(self._mocap)):
            coefficient = self.c2[coefficient_index]
            coefficient_index += 1

            current_root_position = self._mocap[i]["root"][0]
            current_neck_position = self._mocap[i]["neck"][0]

            # Vector from Head to Pelvis in Current Pose
            d_head = (current_root_position[0] - current_neck_position[0], 
                      current_root_position[1] - current_neck_position[1], 
                      current_root_position[2] - current_neck_position[2])

            new_neck_position_x = current_neck_position[0]
            new_neck_position_y = current_neck_position[1]
            new_neck_position_z = current_neck_position[2]

            
            if(self._desired_emotion[0] < 0.0 and self._desired_emotion[2] > 0.0):
                # Usually, high Dominance have the character arch back to elevate the shoulders. For angry (i.e when the pleasure is also low) this is the opposite

                dampening_factor_x = 0.2
                dampening_factor_y = 0.1

                if(coefficient > 1.0):
                    new_neck_position_x += 1.0 * ((coefficient - 1.0) * dampening_factor_x) 
                    new_neck_position_y -= 1.0 * ((coefficient - 1.0) * dampening_factor_y) 
                else:
                    new_neck_position_x -= 1.0 * ((coefficient - 1.0) * dampening_factor_x) 
                    new_neck_position_y += 1.0 * ((coefficient - 1.0) * dampening_factor_y) 

            else:
                if(coefficient > 1.0):
                    dampening_factor_x = 0.05
                    dampening_factor_y = 0.03
                else:
                    dampening_factor_x = 0.2
                    dampening_factor_y = 0.08

                new_neck_position_x -= 1.0 * ((coefficient - 1.0) * dampening_factor_x) 
                new_neck_position_y += 1.0 * ((coefficient - 1.0) * dampening_factor_y) 

            gen_index = i

            print(self.generated_mocap[gen_index]['mocap']['neck'])
            self.generated_mocap[gen_index]['mocap']['neck'] = (
                new_neck_position_x, new_neck_position_y, new_neck_position_z)
            print(self.generated_mocap[gen_index]['mocap']['neck'])
            print()

    def rule_3(self):
        print("\n== RULE 3 - HANDS ==")
        # Move on Vector going from chest to wrist

        coefficient_index = 0
        for i in range(len(self._mocap)):
            #0 C3_HANDS_INDICES
            #1 C3_LEFT_HIPS_INDICES
            #2 C3_RIGHT_HIPS_INDICES
            #3 C3_LEFT_CHEST_INDICES
            #4 C3_RIGHT_CHEST_INDICES

            coefficients = [self.c3_1[coefficient_index], self.c3_2[coefficient_index], self.c3_3[coefficient_index], self.c3_4[coefficient_index], self.c3_5[coefficient_index]]
            coefficient_index += 1

            current_root = self._mocap[i]["root"][0]
            current_chest = self._mocap[i]["chest"][0]
            current_head = self._mocap[i]["neck"][0]

            current_left_hand = self._mocap[i]["left_wrist"][0]
            current_right_hand = self._mocap[i]["right_wrist"][0]


            new_left_hand_position_x = current_left_hand[0]
            new_left_hand_position_y = current_left_hand[1]
            new_left_hand_position_z = current_left_hand[2]

            new_right_hand_position_x = current_right_hand[0]
            new_right_hand_position_y = current_right_hand[1]
            new_right_hand_position_z = current_right_hand[2]

            # LEFT #


            # RIGHT #


            gen_index = i

            #print(self.generated_mocap[gen_index]
            #      ['mocap']['left_wrist'])
            #self.generated_mocap[gen_index]['mocap']['left_wrist'] = (
            #    new_left_hand_position_x, new_left_hand_position_y, new_left_hand_position_z)
            #self.generated_mocap[gen_index]['mocap']['right_wrist'] = (
            #    new_right_hand_position_x, new_right_hand_position_y, new_right_hand_position_z)
            #print(self.generated_mocap[gen_index]
            #      ['mocap']['left_wrist'])
            #print()
        
    def rule_4(self):
        print("\n== RULE 4 - ELBOWS ==")
        # Move on vector going from root to elbow

        coefficient_index = 0
        for i in range(len(self._mocap)):
            coefficient = self.c4[coefficient_index]
            coefficient_index += 1

            current_left_elbow_pos = self._mocap[i]["left_elbow"][0]
            current_right_elbow_pos = self._mocap[i]["right_elbow"][0]
            current_root_position = self._mocap[i]["root"][0]

            # Unit Vectors #
            d_left = np.asarray([current_left_elbow_pos[0] - current_root_position[0], 
                      current_left_elbow_pos[1] - current_root_position[1], 
                      current_left_elbow_pos[2] - current_root_position[2]])
            d_left = d_left / np.linalg.norm(d_left)

            d_right = np.asarray([current_right_elbow_pos[0] - current_root_position[0], 
                      current_right_elbow_pos[1] - current_root_position[1], 
                      current_right_elbow_pos[2] - current_root_position[2]])
            d_right = d_right / np.linalg.norm(d_right)

            # Left #
            new_left_elbow_pos_x = current_left_elbow_pos[0]
            new_left_elbow_pos_y = current_left_elbow_pos[1]
            new_left_elbow_pos_z = current_left_elbow_pos[2]

            # Right #
            new_right_elbow_pos_x = current_right_elbow_pos[0]
            new_right_elbow_pos_y = current_right_elbow_pos[1]
            new_right_elbow_pos_z = current_right_elbow_pos[2]

            if(self._desired_emotion[0] < -0.3 and self._desired_emotion[1] > 0.5 and self._desired_emotion[2] < -0.3): 
                #Usually, high Arousal emotions have broad elbows, but for afraid that is different

                new_left_elbow_pos_x -= d_left[0] * (coefficient-1.0) * 0.5
                new_left_elbow_pos_y -= d_left[1] * (coefficient-1.0) * 0.5
                new_left_elbow_pos_z -= d_left[2] * (coefficient-1.0) * 0.5

                new_right_elbow_pos_x -= d_right[0] * (coefficient-1.0) * 0.5
                new_right_elbow_pos_y -= d_right[1] * (coefficient-1.0) * 0.5
                new_right_elbow_pos_z -= d_right[2] * (coefficient-1.0) * 0.5
            else:
                new_left_elbow_pos_x += d_left[0] * (coefficient-1.0) * 0.5
                new_left_elbow_pos_y += d_left[1] * (coefficient-1.0) * 0.5
                new_left_elbow_pos_z += d_left[2] * (coefficient-1.0) * 0.5

                new_right_elbow_pos_x += d_right[0] * (coefficient-1.0) * 0.5
                new_right_elbow_pos_y += d_right[1] * (coefficient-1.0) * 0.5
                new_right_elbow_pos_z += d_right[2] * (coefficient-1.0) * 0.5

            gen_index = i

            print(self.generated_mocap[gen_index]
                  ['mocap']['left_elbow'])
            self.generated_mocap[gen_index]['mocap']['left_elbow'] = (
                new_left_elbow_pos_x, new_left_elbow_pos_y, new_left_elbow_pos_z)
            print(self.generated_mocap[gen_index]
                  ['mocap']['left_elbow'])
            print()         

            print(self.generated_mocap[gen_index]
                  ['mocap']['right_elbow'])
            self.generated_mocap[gen_index]['mocap']['right_elbow'] = (
                new_right_elbow_pos_x, new_right_elbow_pos_y, new_right_elbow_pos_z)
            print(self.generated_mocap[gen_index]
                  ['mocap']['right_elbow'])
            print()
            

    def rule_1_single(self, frame, coefficient):
        print("\n== RULE 1 SINGLE - PELVIS ==")

        generated_pose = (
                frame["root"][0], frame["root"][1], frame["root"][2])

        return generated_pose

    def rule_2_single(self, frame, coefficient):
        print("\n== RULE 2 SINGLE - NECK ==")
        generated_pose = (
                frame["neck"][0], frame["neck"][1], frame["neck"][2])

        return generated_pose

    def rule_3_single(self, frame, coefficient):
        print("\n== RULE 3 SINGLE - HANDS ==")
        generated_pose_l = (
                frame["left_wrist"][0], frame["left_wrist"][1], frame["left_wrist"][2])

        generated_pose_r = (
                frame["right_wrist"][0], frame["right_wrist"][1], frame["right_wrist"][2])

        return (generated_pose_l, generated_pose_r)

    def rule_4_single(self, frame, coefficient):
        print("\n== RULE 4 SINGLE - ELBOWS ==")
        generated_pose_l = (
                frame["left_elbow"][0], frame["left_elbow"][1], frame["left_elbow"][2])

        generated_pose_r = (
                frame["right_elbow"][0], frame["right_elbow"][1], frame["right_elbow"][2])

        return (generated_pose_l, generated_pose_r)

    """
    def rule_1(self):
        print("\n== RULE 1 - PELVIS ==")
        t_pose_root_height = T_POSE["root"][0][1]
        first = True
        coefficient_index = 0
        for i in range(len(self._mocap)):

            coefficient = self.c1[coefficient_index]
            coefficient_index += 1

            if(coefficient > 1.0):
                t_pose_root_height = T_POSE["root"][0][1]

                current_root_height = self._mocap[i]["root"][0][1]
                new_root_height = current_root_height

                if(current_root_height > t_pose_root_height):
                    new_root_height += (-current_root_height - -t_pose_root_height) * coefficient
                else:
                    new_root_height += (-t_pose_root_height - -current_root_height) * (1.0 - (1.0/coefficient))
                
                if new_root_height > current_root_height + 0.025:
                    new_root_height = current_root_height + 0.025
                elif new_root_height < current_root_height - 0.025:
                    new_root_height = current_root_height - 0.025
                
                gen_index = i
                        
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

                if new_root_height > current_root_height + 0.025:
                    new_root_height = current_root_height + 0.025
                elif new_root_height < current_root_height - 0.025:
                    new_root_height = current_root_height - 0.025

                gen_index = i

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

        dt_head = (-t_pose_root[0] + t_pose_head[0], -t_pose_root[1] +
                   t_pose_head[1], -t_pose_root[2] + t_pose_head[2])

        first = True
        coefficient_index = 0
        for i in range(len(self._mocap)):
            coefficient = self.c2[coefficient_index]
            coefficient_index += 1

            current_root_position = self._mocap[i]["root"][0]
            current_neck_position = self._mocap[i]["neck"][0]

            d_head = (-current_root_position[0] + current_neck_position[0], -current_root_position[1] +
                      current_neck_position[1], -current_root_position[2] + current_neck_position[2])

            new_neck_position_x = current_neck_position[0]
            new_neck_position_y = current_neck_position[1]
            new_neck_position_z = current_neck_position[2]

            new_neck_position_x += dt_head[0] - \
                ((dt_head[0]-d_head[0])/coefficient) - d_head[0]
            new_neck_position_y += dt_head[1] - \
                ((dt_head[1]-d_head[1])/coefficient) - d_head[1]
            new_neck_position_z += dt_head[2] - \
                ((dt_head[2]-d_head[2])/coefficient) - d_head[2]

            gen_index = i

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
            #0 C3_HANDS_INDICES
            #1 C3_LEFT_HIPS_INDICES
            #2 C3_RIGHT_HIPS_INDICES
            #3 C3_LEFT_CHEST_INDICES
            #4 C3_RIGHT_CHEST_INDICES

            coefficients = [self.c3_1[coefficient_index], self.c3_2[coefficient_index], self.c3_3[coefficient_index], self.c3_4[coefficient_index], self.c3_5[coefficient_index]]
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

            dl_right = (current_right_hand[0] - current_left_hand[0], current_right_hand[1] -
                       current_left_hand[1], current_right_hand[2] - current_left_hand[2])
            dr_left = (current_left_hand[0] - current_right_hand[0], current_left_hand[1] -
                        current_right_hand[1], current_left_hand[2] - current_right_hand[2])

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

            # LEFT #
            new_left_hand_position_x += (dl_right[0] * (coefficients[0] - 1.0) + dl_ground[0] * (coefficients[0] - 1.0) + dl_chest[0] * (
                coefficients[3] - 1.0) + dl_root[0] * (coefficients[1] - 1.0))/4.0
            new_left_hand_position_y += (dl_right[1] * (coefficients[0] - 1.0) + dl_ground[1] * (coefficients[0] - 1.0) + dl_chest[1] * (
                coefficients[3] - 1.0) + dl_root[1] * (coefficients[1] - 1.0))/4.0
            new_left_hand_position_z += (dl_right[1] * (coefficients[0] - 1.0) + dl_ground[1] * (coefficients[0] - 1.0) + dl_chest[1] * (
                coefficients[3] - 1.0) + dl_root[2] * (coefficients[1] - 1.0))/4.0

            # RIGHT #
            new_right_hand_position_x += (dr_left[0] * (coefficients[0] - 1.0) + dr_ground[0] * (coefficients[0] - 1.0) + dr_chest[0] * (
                coefficients[4] - 1.0) + dr_root[0] * (coefficients[2] - 1.0))/4.0
            new_right_hand_position_y += (dr_left[1] * (coefficients[0] - 1.0) + dr_ground[1] * (coefficients[0] - 1.0) + dr_chest[1] * (
                coefficients[4] - 1.0) + dr_root[1] * (coefficients[2] - 1.0))/4.0
            new_right_hand_position_z += (dr_left[2] * (coefficients[0] - 1.0) + dr_ground[2] * (coefficients[0] - 1.0) + dr_chest[2] * (
                coefficients[4] - 1.0) + dr_root[2] * (coefficients[2] - 1.0))/4.0

            gen_index = i

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
        print("\n== RULE 4 - ELBOWS ==")
        t_pose_left_shoulder_height = T_POSE["left_shoulder"][0][1]
        t_pose_right_shoulder_height = T_POSE["right_shoulder"][0][1]
        first = True
        coefficient_index = 0
        for i in range(len(self._mocap)):

            coefficients = [self.c4_1[coefficient_index], self.c4_2[coefficient_index]]
            coefficient_index += 1

            # LEFT
            if(coefficients[0] > 1.0):
                current_left_shoulder_height = self._mocap[i]["left_shoulder"][0][1]
                new_left_shoulder_height = current_left_shoulder_height

                if(current_left_shoulder_height > t_pose_left_shoulder_height):
                    new_left_shoulder_height += (-current_left_shoulder_height - -t_pose_left_shoulder_height) * coefficients[0]
                else:
                    new_left_shoulder_height += (-t_pose_left_shoulder_height - -current_left_shoulder_height) * (1.0 - (1.0/coefficients[0]))
                
                if new_left_shoulder_height > current_left_shoulder_height + 0.05:
                    new_left_shoulder_height = current_left_shoulder_height + 0.05
                elif new_left_shoulder_height < current_left_shoulder_height - 0.05:
                    new_left_shoulder_height = current_left_shoulder_height - 0.05
                
                gen_index = i
                        
                #print("Original:" + str(self.generated_mocap[gen_index]['mocap']['left_shoulder']))

                #temp = self.generated_mocap[gen_index]['mocap']['left_shoulder']
                #self.generated_mocap[gen_index]['mocap']['left_shoulder'] = (
                #    temp[0], new_left_shoulder_height, temp[2])

                #print("Synthesized:" + str(self.generated_mocap[gen_index]['mocap']['left_shoulder']))
                #print()

            else:
                current_left_shoulder_height = self._mocap[i]["left_shoulder"][0][1]
                new_left_shoulder_height = current_left_shoulder_height

                new_left_shoulder_height += (-current_left_shoulder_height *
                                    (coefficients[0] - 1.0))

                if new_left_shoulder_height > current_left_shoulder_height + 0.05:
                    new_left_shoulder_height = current_left_shoulder_height + 0.05
                elif new_left_shoulder_height < current_left_shoulder_height - 0.05:
                    new_left_shoulder_height = current_left_shoulder_height - 0.05

                gen_index = i

                #print("Original:" + str(self.generated_mocap[gen_index]['mocap']['left_shoulder']))

                #temp = self.generated_mocap[gen_index]['mocap']['left_shoulder']
                #self.generated_mocap[gen_index]['mocap']['left_shoulder'] = (
                #    temp[0], new_left_shoulder_height, temp[2])

                #print("Synthesized:" + str(self.generated_mocap[gen_index]['mocap']['left_shoulder']))
                #print()


            
            # RIGHT
            if(coefficients[1] > 1.0):
                current_right_shoulder_height = self._mocap[i]["right_shoulder"][0][1]
                new_right_shoulder_height = current_right_shoulder_height

                if(current_right_shoulder_height > t_pose_right_shoulder_height):
                    new_right_shoulder_height += (-current_right_shoulder_height - -t_pose_right_shoulder_height) * coefficients[1]
                else:
                    new_right_shoulder_height += (-t_pose_right_shoulder_height - -current_right_shoulder_height) * (1.0 - (1.0/coefficients[1]))
                
                if new_right_shoulder_height > current_right_shoulder_height + 0.05:
                    new_right_shoulder_height = current_right_shoulder_height + 0.05
                elif new_right_shoulder_height < current_right_shoulder_height - 0.05:
                    new_right_shoulder_height = current_right_shoulder_height - 0.05
                
                gen_index = i
                        
                #print("Original:" + str(self.generated_mocap[gen_index]['mocap']['right_shoulder']))

                #temp = self.generated_mocap[gen_index]['mocap']['right_shoulder']
                #self.generated_mocap[gen_index]['mocap']['right_shoulder'] = (
                #    temp[0], new_right_shoulder_height, temp[2])

                #print("Synthesized:" + str(self.generated_mocap[gen_index]['mocap']['right_shoulder']))
                #print()

            else:
                current_right_shoulder_height = self._mocap[i]["right_shoulder"][0][1]
                new_right_shoulder_height = current_right_shoulder_height
                
                new_right_shoulder_height += (-current_right_shoulder_height *
                                    (coefficients[1] - 1.0))

                if new_right_shoulder_height > current_right_shoulder_height + 0.05:
                    new_right_shoulder_height = current_right_shoulder_height + 0.05
                elif new_right_shoulder_height < current_right_shoulder_height - 0.05:
                    new_right_shoulder_height = current_right_shoulder_height - 0.05

                gen_index = i

                #print("Original:" + str(self.generated_mocap[gen_index]['mocap']['right_shoulder']))

                #temp = self.generated_mocap[gen_index]['mocap']['right_shoulder']
                #self.generated_mocap[gen_index]['mocap']['right_shoulder'] = (
                #    temp[0], new_right_shoulder_height, temp[2])

                #print("Synthesized:" + str(self.generated_mocap[gen_index]['mocap']['right_shoulder']))
                #print()

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
# c1 - Pelvis Height
# c2 - Head Height (maybe shoulders instead?)
# c3 - Hands
# c4 - Elbow Height

#00          c3         "max_hand_distance",
#01          c3         "avg_l_hand_hip_distance",
#02          c3         "avg_r_hand_hip_distance",
#03          --         "max_stride_length",
#04          c3         "avg_l_hand_chest_distance",
#05          c3         "avg_r_hand_chest_distance",
#06          c5         "avg_l_elbow_hip_distance",
#07          c5         "avg_r_elbow_hip_distance",
#08          c1         "avg_chest_pelvis_distance",
#09          c2         "avg_neck_chest_distance",
#          
#10          c1, c2,c3  "avg_total_body_volume",
#11          c1,c2,c3   "avg_lower_body_volume",
#12          c2,c3      "avg_upper_body_volume",
#          
#13          c3         "avg_triangle_area_hands_neck",
#14          c1         "avg_triangle_area_feet_hips",
#          
#15          --         "l_hand_speed",
#16          --         "r_hand_speed",
#17          --         "l_foot_speed",
#18          --         "r_foot_speed",
#19          --         "neck_speed",
#          
#20          --         "l_hand_acceleration_magnitude",
#21          --         "r_hand_acceleration_magnitude",
#22          --         "l_foot_acceleration_magnitude",
#23          --         "r_foot_acceleration_magnitude",
#24          --         "neck_acceleration_magnitude",                    ]


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
