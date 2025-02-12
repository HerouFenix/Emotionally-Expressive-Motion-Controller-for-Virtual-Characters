import math
import numpy as np
import os.path
import os
import pybullet as p1
import pandas as pd


import pandas as pd

import xgboost as xgb

import tensorflow as tf
from tensorflow import keras


xgb.set_config(verbosity=0)
from scipy.optimize import minimize

#from gui_manager import GUIManager

PRESET_EMOTIONS = {
    # NORMAL
    (0.05, -0.05, 0.0): [ 0.5186799,   0.46649932,  0.15772843,  0.64237505,  0.37424588,  0.18964156,
   0.12902201,  0.24415847,  0.23615103,  0.22389398,  0.33483888,  0.09003431,
   0.1015784 ,  0.07853313,  0.2298812 , -0.16199612, -0.15539091, -0.20247241,
  -0.12584846, -0.15541993,  0.02993792,  0.02793053,  0.03596406,  0.04410392,
   0.03035523],
    
    # TIRED
    (0.1, -0.7, -0.2): [ 0.4450052,   0.30267617,  0.44533758,  0.29547794,  0.23564595,  0.35828984,
   0.11714197,  0.13302721,  0.236151  ,  0.22389399,  0.20719219,  0.07022425,
   0.06840402,  0.06758865,  0.08148723, -0.13730852, -0.13381774, -0.05919065,
  -0.14447553, -0.1255996 ,  0.01493876,  0.02008506,  0.00624596,  0.01632991,
   0.01687064],
    
    # TIRED 2
    (0.1, -0.75, -0.25): [ 0.4450052,   0.30267617,  0.44533758,  0.29547794,  0.23564595,  0.35828984,
   0.11714197,  0.13302721,  0.236151  ,  0.22389399,  0.20719219,  0.07022425,
   0.06840402,  0.06758865,  0.08148723, -0.13730852, -0.13381774, -0.05919065,
  -0.14447553, -0.1255996 ,  0.01493876,  0.02008506,  0.00624596,  0.01632991,
   0.01687064],
    
    # EXHAUSTED
    (-0.1, -0.6, -0.15): [ 0.4450052,   0.30267617,  0.44533758,  0.29547794,  0.23564595,  0.35828984,
   0.11714197,  0.13302721,  0.236151  ,  0.22389399,  0.20719219,  0.07022425,
   0.06840402,  0.06758865,  0.08148723, -0.13730852, -0.13381774, -0.05919065,
  -0.14447553, -0.1255996 ,  0.01493876,  0.02008506,  0.00624596,  0.01632991,
   0.01687064],
    
    # ANGRY
    (-0.5, 0.8, 0.9): [ 0.4849567,   0.43509817,  0.40199345,  0.72839327,  0.33868578,  0.19559014,
   0.41532856,  0.26116835,  0.23615097,  0.22389401,  0.4839097 ,  0.10084644,
   0.0859726 ,  0.02596105,  0.21177863, -0.2379167 , -0.21990552, -0.16867742,
  -0.28235362, -0.21497578,  0.02257726,  0.07411671,  0.0837294 ,  0.01054426,
   0.02439681],

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
    (0.25, 0.15, 0.4): [ 0.72487293,  0.35887094,  0.42138989,  0.78525169,  0.31323459,  0.41962872,
   0.3456292 ,  0.36508856,  0.236151  ,  0.22389399,  0.64049156,  0.14345473,
   0.15673073,  0.10815576,  0.27122958, -0.12987   , -0.15814968, -0.13852751,
  -0.14657472, -0.14309478,  0.02549621,  0.01590126,  0.01748402,  0.03745181,
   0.01475212],
    
    # CONFIDENT 3
    (0.3, 0.4, 0.6): [ 0.72487293,  0.35887094,  0.42138989,  0.78525169,  0.31323459,  0.41962872,
   0.3456292 ,  0.36508856,  0.236151  ,  0.22389399,  0.64049156,  0.14345473,
   0.15673073,  0.10815576,  0.27122958, -0.12987   , -0.15814968, -0.13852751,
  -0.14657472, -0.14309478,  0.02549621,  0.01590126,  0.01748402,  0.03745181,
   0.01475212],
    
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
    (-0.5, 0.7, 0.9): [ 0.4450052,   0.30267617,  0.44533758,  0.29547794,  0.23564595,  0.35828984,
   0.11714197,  0.13302721,  0.236151  ,  0.22389399,  0.20719219,  0.07022425,
   0.06840402,  0.06758865,  0.08148723, -0.13730852, -0.13381774, -0.05919065,
  -0.14447553, -0.1255996 ,  0.01493876,  0.02008506,  0.00624596,  0.01632991,
   0.01687064],

    # HAPPY
    (0.8, 0.5, 0.15): [ 0.87840172,  0.40800089,  0.69014457,  0.60317439,  0.4222134,   0.57671535,
   0.22764189,  0.396286  ,  0.236151  ,  0.223894  ,  0.5703048 ,  0.1081096,
   0.14953151,  0.14243552,  0.22793017, -0.09510082, -0.15857715, -0.06167734,
  -0.07744886, -0.07451435,  0.05870626,  0.06389066,  0.03951913,  0.04674497,
   0.04283095],

    # SAD 2
    (-0.6, -0.3, -0.3): [ 0.48203249,  0.30429993,  0.32106327,  0.3275376,   0.17515474,  0.30992711,
   0.25770942,  0.38230947,  0.23615101,  0.223894  ,  0.35171322,  0.08729186,
   0.05838501,  0.05864761,  0.09994176, -0.07459375, -0.13680961, -0.13222726,
  -0.05183191, -0.08656714,  0.0209568 ,  0.02331046,  0.00706071,  0.01917414,
   0.02123904],

    # AFRAID 2
    (-0.5, 0.7, -0.8): [ 2.08340828e-01,  2.64129062e-01,  2.27426258e-01,  2.30559942e-01,
   1.28220619e-01,  1.26505977e-01,  2.62098755e-01,  3.14412678e-01,
   2.36151019e-01,  2.23893942e-01,  2.06627529e-01,  5.68116833e-02,
   3.86782264e-02,  2.64402409e-02,  9.19677408e-02, -1.91390101e-02,
  -2.03380232e-02, -1.51719401e-02, -1.05658747e-02, -1.84225226e-02,
   9.78201230e-04,  7.71277062e-04,  1.90137609e-03,  1.32488543e-03,
   2.00411355e-04],
}

PRESET_EMOTIONS_2 = {
    (0.05, -0.05, 0.0): [ 0.64861105,  0.49560863,  0.14687989,  0.24520622,  0.42659557,
        0.25471957,  0.26435806,  0.10474267,  0.23615097,  0.22389403,
        0.34194623,  0.09833431,  0.05393501,  0.1057551 ,  0.05788131,
       -0.20031017, -0.13354014, -0.08614763, -0.17877042, -0.13467219,
        0.20031017,  0.13354014,  0.08614763,  0.17877042,  0.13467219], 
        (0.1, -0.7, -0.2): [ 0.58383973,  0.31521119,  0.26822572,  0.53903728,  0.39098912,
        0.3841018 ,  0.27826615,  0.24514007,  0.23615102,  0.22389397,
        0.33500452,  0.07490563,  0.11688627,  0.13879168,  0.19007616,
       -0.04769258, -0.09320253, -0.09363547, -0.0567676 , -0.06321911,
        0.01412272,  0.00348533,  0.01083853,  0.00802987,  0.00863552], 
        (0.1, -0.75, -0.25): [ 0.45834358,  0.23786188,  0.29241195,  0.70583622,  0.21903112,
        0.25616385,  0.06234959,  0.2843048 ,  0.23615123,  0.22389381,
        0.3789303 ,  0.08134197,  0.12621986,  0.0686663 ,  0.24225937,
       -0.18842486, -0.15149203, -0.22261945, -0.12004481, -0.15933352,
        0.18842486,  0.15149203,  0.22261945,  0.12004481,  0.15933352], 
        (-0.1, -0.6, -0.15): [ 0.53039867,  0.28257401,  0.25054556,  0.172366  ,  0.3817522 ,
        0.3646494 ,  0.27215122,  0.25406212,  0.23615099,  0.22389402,
        0.16580799,  0.02874755,  0.04524075,  0.12880294,  0.05966161,
       -0.02495014, -0.04606   , -0.08055927, -0.01948031, -0.0385244 ,
        0.00197816,  0.00236151,  0.0062883 ,  0.00170289,  0.00184331], 
        (-0.5, 0.8, 0.9): [ 0.86056251,  0.64542863,  0.44947413,  0.70387046,  0.57760021,
        0.30747418,  0.42347914,  0.28382269,  0.23615098,  0.22389404,
        0.7469025 ,  0.19989731,  0.08931127,  0.05936749,  0.25463615,
       -0.10656026, -0.14082686, -0.11884943, -0.10147461, -0.14052722,
        0.03080482,  0.03851178,  0.04781011,  0.01867437,  0.01519821], 
        (0.8, 0.5, 0.15): [ 0.49245619,  0.36812286,  0.09931385,  0.27794179,  0.35438074,
        0.22374433,  0.21296117,  0.09576121,  0.23615091,  0.22389397,
        0.28035424,  0.07474504,  0.0559551 ,  0.08659375,  0.07148009,
       -0.18723606, -0.11413441, -0.07967259, -0.17750639, -0.1280146 ,
        0.18723606,  0.11413441,  0.07967259,  0.17750639,  0.1280146 ], 
        (0.6, 0.4, 0.1): [ 0.33840718,  0.22201569,  0.10957601,  0.24691763,  0.29965145,
        0.29641824,  0.11279353,  0.13917627,  0.23615095,  0.22389409,
        0.16380432,  0.04852431,  0.05370823,  0.06749195,  0.07018122,
       -0.05279276, -0.145224  , -0.09524675, -0.03077059, -0.06561986,
        0.02198676,  0.02815459,  0.01774675,  0.01805181,  0.00222353], 
        (-0.6, -0.4, -0.3): [ 0.58873841,  0.29738092,  0.33641582,  0.23611636,  0.39255141,
        0.40247339,  0.26841262,  0.25437805,  0.23615101,  0.22389399,
        0.24940353,  0.05807406,  0.06005707,  0.13893916,  0.07807057,
       -0.02662492, -0.08019933, -0.11299456, -0.01955708, -0.05149339,
        0.02662492,  0.08019933,  0.11299456,  0.01955708,  0.05149339], 
        (0.4, 0.2, 0.35): [ 0.69438987,  0.37840008,  0.39968552,  0.18058271,  0.31787106,
        0.42388732,  0.3303118 ,  0.40078659,  0.23615099,  0.22389404,
        0.3936062 ,  0.12330935,  0.05188695,  0.11105894,  0.04643896,
       -0.14738379, -0.09300292, -0.07132587, -0.14400662, -0.11419529,
        0.02492978,  0.03829397,  0.02034157,  0.0009977 ,  0.01735854], 
        (0.3, 0.3, 0.9): [ 0.59417624,  0.33119177,  0.26035112,  0.21796067,  0.32150634,
        0.34090365,  0.27408675,  0.31139454,  0.23615102,  0.2238941 ,
        0.33802211,  0.06231024,  0.05283739,  0.10975461,  0.03413105,
       -0.18107179, -0.11453219, -0.07736814, -0.16755109, -0.12402669,
        0.18107179,  0.11453219,  0.07736814,  0.16755109,  0.12402669], 
        (0.25, 0.15, 0.4): [ 0.53752993,  0.40478774,  0.44096886,  0.45892374,  0.37931977,
        0.23670061,  0.44773889,  0.24462895,  0.23615098,  0.22389404,
        0.42467847,  0.07907314,  0.06759834,  0.03936626,  0.16414483,
       -0.09406014, -0.08451607, -0.11034161, -0.05071193, -0.06946846,
        0.09406014,  0.08451607,  0.11034161,  0.05071193,  0.06946846], 
        (0.3, 0.4, 0.6): [ 0.78157806,  0.39862087,  0.41273468,  0.52793637,  0.37785825,
        0.43724336,  0.34841091,  0.35716291,  0.23615098,  0.22389396,
        0.64274743,  0.12528543,  0.0883277 ,  0.13362552,  0.17660202,
       -0.14491051, -0.1410848 , -0.12545425, -0.20998618, -0.14879689,
        0.02235624,  0.03590835,  0.02294979,  0.02423578,  0.02472427], 
        (-0.6, 0.7, -0.8): [ 0.23015596,  0.36543437,  0.33017059,  0.29385905,  0.21461348,
        0.12564734,  0.28071408,  0.22933138,  0.23615095,  0.2238941 ,
        0.2512212 ,  0.05397538,  0.04983733,  0.01926071,  0.07666902,
       -0.11703679, -0.1071213 , -0.0593118 , -0.14730672, -0.10082388,
        0.02901676,  0.02638158,  0.01723556,  0.00563751,  0.02696801], 
        (0.1, 0.6, 0.4): [ 0.89305512,  0.41581097,  0.47427136,  0.34804665,  0.43294892,
        0.50026181,  0.33820047,  0.36402987,  0.23615101,  0.223894  ,
        0.45284732,  0.16453248,  0.08604966,  0.16798674,  0.12018277,
       -0.08099741, -0.02859092, -0.02427842, -0.12168651, -0.04501502,
        0.08099741,  0.02859092,  0.02427842,  0.12168651,  0.04501502]
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

#00          c3_hips, c3_chest                      "max_hand_distance",
#01          c3_hips                                "avg_l_hand_hip_distance",
#02          c3_hips                                "avg_r_hand_hip_distance",
#03          c5                                     "max_stride_length",
#04          c3_chest                               "avg_l_hand_chest_distance",
#05          c3_chest                               "avg_r_hand_chest_distance",
#06          c4                                     "avg_l_elbow_hip_distance",
#07          c4                                     "avg_r_elbow_hip_distance",
#08          c1, c2                                 "avg_chest_pelvis_distance",
#09          c6                                     "avg_neck_chest_distance",
#          
#10          c1, c2, c3_hips, c3_chest, c4, c5, c6  "avg_total_body_volume",
#11          c1, c5                                 "avg_lower_body_volume",
#12          c2, c3_hips, c3_chest, c4, c6          "avg_upper_body_volume",
#          
#13          c3_hips, c3_chest                      "avg_triangle_area_hands_neck",
#14          c1,c5                                  "avg_triangle_area_feet_hips",

C1_INDICES = [8, 10, 11, 14] # Root
C2_INDICES = [8, 10, 12] # Chest

C3_INDICES = [0,1,2,4,5, 10, 12,13] # L/R Hand [DEPRECATED]
C3_HIPS_INDICES = [0, 1, 2, 10, 12, 13] # Hand-Hips
C3_CHEST_INDICES = [0, 4, 5, 10, 12, 13] # Hand-Chest

C4_INDICES = [6,7,10,12] # Elbows
C4_LEFT_INDICES = [6] # L-Elbow [DEPRECATED]
C4_RIGHT_INDICES = [7] # R-Elbow [DEPRECATED]

C5_INDICES = [3,10, 11, 14] # Feet

C6_INDICES = [9,10, 12] # Chest Z-Rotation


class MotionSynthesizer():
    def __init__(self,models=""):
        # Initializer with no files
        self.c1 = [1.0]
        self.c2 = [1.0]

        self.c3 = [1.0]
        self.c3_1 = [1.0] #C3_HANDS_HIPS_INDICES
        self.c3_2 = [1.0] #C3_HANDS_CHEST_INDICES

        self.c4 = [1.0]

        self.c5 = [1.0] #C5

        self.c6 = [1.0]

        self.current_reference_features = []
        self.current_features = []

        self._mocap = []
        self.generated_mocap = []

        self._lma = []

        self._reference = []

        if(models=="direct" or models=="DIRECT" or models == "xgb" or models == "XGB"):
            self._models = {}
            self._models_type = models

            #for filename in os.listdir("../motion_synthesizer/models/bandai/xgb/"):
            #    f = os.path.join("../motion_synthesizer/models/bandai/xgb/", filename)
            for filename in os.listdir("../motion_synthesizer/models/bandai/xgb/"):
                f = os.path.join("../motion_synthesizer/models/bandai/xgb/", filename)
                if os.path.isfile(f):
                    print(filename)
                    model = xgb.XGBRegressor(verbosity=0)
                    model.load_model(f)
                    self._models[filename.split(".")[0]] = model

            #print(self._models)
        
        elif(models=="ae" or models=="AE"):
            self._models = {"ae": None, "xgb": {}}
            self._models_type = models

            #ae_model = keras.models.load_model('../motion_synthesizer/models/bandai/ae/autoencoder_5/')
            ae_model = keras.models.load_model('../motion_synthesizer/models/bandai/ae/autoencoder_5/')
            self._models["ae"] = ae_model

            #for filename in os.listdir("../motion_synthesizer/models/bandai/ae/"):
            #    f = os.path.join("../motion_synthesizer/models/bandai/ae/", filename)
            for filename in os.listdir("../motion_synthesizer/models/bandai/ae/"):
                f = os.path.join("../motion_synthesizer/models/bandai/ae/", filename)
                if os.path.isfile(f):
                    print(filename)
                    model = xgb.XGBRegressor(verbosity=0)
                    model.load_model(f)
                    self._models["xgb"][filename.split(".")[0]] = model

            #print(self._models)
        else:
            self._models = {}
            self._models_type = models

        self._desired_emotion = np.asarray([0.0, 0.0, 0.0])


    def reset(self):
        self.c1 = [1.0]
        self.c2 = [1.0]

        self.c3 = [1.0]
        self.c3_1 = [1.0] #C3_HANDS_HIPS_INDICES
        self.c3_2 = [1.0] #C3_HANDS_CHEST_INDICES

        self.c4 = [1.0]

        self.c5 = [1.0] #C5

        self.c6 = [1.0]

        self.current_reference_features = []
        self.current_features = []

        self._mocap = []
        self.generated_mocap = []

        self._lma = []

        self._reference = []

        self._desired_emotion = np.asarray([0.0, 0.0, 0.0])

    def reset_coefficients(self):
        self.c1 = [1.0]
        self.c2 = [1.0]

        self.c3 = [1.0]
        self.c3_1 = [1.0] #C3_HANDS_HIPS_INDICES
        self.c3_2 = [1.0] #C3_HANDS_CHEST_INDICES

        self.c4 = [1.0]

        self.c5 = [1.0] #C5

        self.c6 = [1.0]

        self.generated_mocap = []
        self._reference = []


    def set_current_mocap(self, mocap):
        self._mocap = []
        self.generated_mocap = []
        
        i = 1
        for frame in mocap:
            frame = frame.copy()
            self._mocap.append(frame) # This already ignores the first frame, which is why we start i at 1
            gen = {"root": frame["root"][0], "neck": frame["neck"][0], "left_wrist":frame["left_wrist"][0], "right_wrist":frame["right_wrist"][0], "left_elbow":frame["left_elbow"][0], "right_elbow":frame["right_elbow"][0], "left_ankle":frame["left_ankle"][0], "right_ankle":frame["right_ankle"][0]}
            orn = {"root": frame["root"][1], "neck": frame["neck"][1], "left_wrist":frame["left_wrist"][1], "right_wrist":frame["right_wrist"][1], "left_elbow":frame["left_elbow"][1], "right_elbow":frame["right_elbow"][1], "left_ankle":frame["left_ankle"][1], "right_ankle":frame["right_ankle"][1]}
            
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
            self._reference = PRESET_EMOTIONS[(0.3, 0.3, 0.9)] # Confident
            self._desired_emotion = np.asarray([0.3, 0.3, 0.9])

            #self._reference = PRESET_EMOTIONS[(-0.5, 0.8, 0.9)] # Angry
            #self._desired_emotion = np.asarray([-0.5, 0.8, 0.9])

            #self._reference = PRESET_EMOTIONS[(-0.6, 0.7, -0.8)] # Afraid
            #self._desired_emotion = np.asarray([-0.6, 0.7, -0.8])

        else:
            self._reference = lma


    def _find_closest_emotion(self, pad):
        p, a, d = pad
        #dist = lambda key: (p - EMOTION_COORDINATES[key][0]) ** 2 + (a - EMOTION_COORDINATES[key][1]) ** 2 + (d - EMOTION_COORDINATES[key][2]) ** 2
        dist = lambda key: (p - key[0]) ** 2 + (a - key[1]) ** 2 + (d - key[2]) ** 2
        closest_coordinates = min(PRESET_EMOTIONS, key=dist)

        distance = (p - closest_coordinates[0]) ** 2 + (a - closest_coordinates[1]) ** 2 + (d - closest_coordinates[2]) ** 2

        return closest_coordinates, distance
    
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




        # Check if we have a "close enough" preset emotion
        emotion, dist = self._find_closest_emotion(pad)
        if(dist <= 0.01):
            lma = PRESET_EMOTIONS[emotion]
            pad = np.asarray([emotion[0], emotion[1], emotion[2]])
            self._desired_emotion = pad
        elif(self._models_type == "direct" or self._models_type == "DIRECT" or self._models_type == "xgb" or self._models_type == "XGB"):
            pad_predict = np.asarray([[pad[0], pad[1], pad[2]]])
            
            pad = np.asarray([pad[0], pad[1], pad[2]])
            self._desired_emotion = pad

            for feature in pad_order:
                lma.append(self._models[feature].predict(pad_predict)[0])

        elif(self._models_type == "ae" or self._models_type == "AE"):
            pad_predict = np.asarray([[pad[0], pad[1], pad[2]]])
            
            pad = np.asarray([pad[0], pad[1], pad[2]])
            self._desired_emotion = pad

            latent_order = ["bandai_pad2l1_model",
                     "bandai_pad2l2_model",
                     "bandai_pad2l3_model",
                     "bandai_pad2l4_model",
                     "bandai_pad2l5_model"]
            latent_space = []



            for latent_feature in latent_order:
                latent_space.append(self._models["xgb"][latent_feature].predict(pad_predict)[0])

            latent_space = np.asarray([latent_space])
            lma = self._models["ae"].decoder.predict(latent_space)[0]
            
        else:
            lma = PRESET_EMOTIONS[emotion]
            pad = np.asarray([emotion[0], emotion[1], emotion[2]])
            self._desired_emotion = pad
            print(pad)
        
        self._reference = lma
        #print(self._reference)


    def compute_coefficients(self):
        self.compute_coefficient(1)
        self.compute_coefficient(2)
        
        self.compute_coefficient(3.1)
        self.compute_coefficient(3.2)
        
        self.compute_coefficient(4)
        
        self.compute_coefficient(5)

        self.compute_coefficient(6)

    def get_motion_changes(self):
        self.rule_1()
        self.rule_2()
        self.rule_3()
        self.rule_4()
        self.rule_5()
        self.rule_6()

        return self.generated_mocap

    def convert_single_frame(self, frame, counter, pose):
        # Get the coefficients of the closest key frame and compute the changes of this single frame alone
        #closest_index = int((counter/self._frame_worth)/self._extraction_framerate)
        closest_index = int(counter/5)

        #if(closest_index >= len(self.c1)):
        #    closest_index = len(self.c1)-1

        frame_c1 = self.c1#[closest_index]
        frame_c2 = self.c2#[closest_index]
        frame_c3_1 = self.c3_1#[closest_index]
        frame_c3_2 = self.c3_2#[closest_index]
        frame_c4 = self.c4#[closest_index]
        frame_c5 = self.c5#[closest_index]
        frame_c6 = self.c6

        generated = {"mocap": {"root": [], "neck": [], "left_wrist": [], "right_wrist":[], "left_elbow":[], "right_elbow":[], "left_ankle": [], "right_ankle": []}, "orn": {"neck": []}}
        root = self.rule_1_single(frame, frame_c1)
        neck = self.rule_2_single(frame, frame_c2)
        left_wrist, right_wrist = self.rule_3_single(frame, frame_c3_1, frame_c3_2)
        left_elbow, right_elbow = self.rule_4_single(frame, frame_c4)
        left_ankle, right_ankle = self.rule_5_single(frame, frame_c5)
        neck_rotation = self.rule_6_single(pose, frame_c6)

        generated["mocap"]["root"] = root
        generated["mocap"]["neck"] = neck
        generated["mocap"]["left_wrist"] = left_wrist
        generated["mocap"]["right_wrist"] = right_wrist
        generated["mocap"]["left_elbow"] = left_elbow
        generated["mocap"]["right_elbow"] = right_elbow
        generated["mocap"]["left_ankle"] = left_ankle
        generated["mocap"]["right_ankle"] = right_ankle
        generated["orn"]["neck"] = neck_rotation

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
            print("== COMPUTING COEFFICIENT C3 - HANDS-HIPS ==")
            feature_index = C3_HIPS_INDICES
        elif(coefficient_number == 3.2):
            print("== COMPUTING COEFFICIENT C3 - HANDS-CHEST ==")
            feature_index = C3_CHEST_INDICES

        elif(coefficient_number == 4):
            print("== COMPUTING COEFFICIENT C4 - ELBOWS ==")
            feature_index = C4_INDICES
        elif(coefficient_number == 4.1):
            print("== COMPUTING COEFFICIENT C4 - LEFT ELBOW ==")
            feature_index = C4_LEFT_INDICES
        elif(coefficient_number == 4.2):
            print("== COMPUTING COEFFICIENT C4 - RIGHT ELBOW ==")
            feature_index = C4_RIGHT_INDICES

        elif(coefficient_number == 5):
            print("== COMPUTING COEFFICIENT C5 - FEET ==")
            feature_index = C5_INDICES

        elif(coefficient_number == 6):
            print("== COMPUTING COEFFICIENT C6 - NECK ROTATION ==")
            feature_index = C6_INDICES

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

        coefficient = 1.0 #[1.0] * len(self.current_features)

        res = minimize(self.compute_sse,
                       coefficient,
                       method='Powell',
                       #callback = self.minimize_callback,
                       options={'maxiter': 50000, 'disp': False}) # TODO: REDUCE MAXITER

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
            
        elif(coefficient_number == 4):
            self.c4 = coefficient
        elif(coefficient_number == 4.1):
            self.c4_1 = coefficient
        elif(coefficient_number == 4.2):
            self.c4_2 = coefficient

        elif(coefficient_number == 5):
            self.c5 = coefficient

        elif(coefficient_number == 6):
            self.c6 = coefficient

    def compute_norms(self, reference, current_features, coefficient):
        norms = []

        for i in range(len(current_features)):
            norms.append(np.linalg.norm(
                reference - (current_features[i] * coefficient)) ** 2)

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
            coefficient = self.c1#[coefficient_index]
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
            new_root_height += 1.0 * ((coefficient - 1.0) * 0.08) #0.1 -> change weight

                
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
            coefficient = self.c2#[coefficient_index]
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

            
            #if(self._desired_emotion[0] < 0.0 and self._desired_emotion[2] > 0.0):
            #    # Usually, high Dominance have the character arch back to elevate the shoulders. For angry (i.e when the pleasure is also low) this is the opposite
            #
            #    dampening_factor_x = 0.25
            #    dampening_factor_y = 0.15
            #
            #    if(coefficient > 1.0):
            #        new_neck_position_x += 1.0 * (((coefficient+0.2) - 1.0) * dampening_factor_x) 
            #        new_neck_position_y -= 1.0 * (((coefficient+0.2) - 1.0) * dampening_factor_y) 
            #    else:
            #        new_neck_position_x -= 1.0 * (((coefficient-0.2) - 1.0) * dampening_factor_x) 
            #        new_neck_position_y += 1.0 * (((coefficient-0.2) - 1.0) * dampening_factor_y) 
            #
            #elif(self._desired_emotion[0] < 0.0 and self._desired_emotion[1] < 0.0 and self._desired_emotion[2] < 0.0):
            #    if(coefficient > 1.0):
            #        dampening_factor_x = 0.3
            #        dampening_factor_y = 0.2
            #    else:
            #        dampening_factor_x = 0.3
            #        dampening_factor_y = 0.2
            #
            #    new_neck_position_x -= 1.0 * ((coefficient - 1.0) * dampening_factor_x) 
            #    new_neck_position_y += 1.0 * ((coefficient - 1.0) * dampening_factor_y) 
            #
            #else:
            #    if(coefficient > 1.0):
            #        dampening_factor_x = 0.25
            #        dampening_factor_y = 0.15
            #    else:
            #        dampening_factor_x = 0.25
            #        dampening_factor_y = 0.15
            #   new_neck_position_x -= 1.0 * ((coefficient - 1.0) * change_weight_factor_x) 
            #    new_neck_position_y += 1.0 * ((coefficient - 1.0) * change_weight_factor_y) 

            if(self._desired_emotion[0] < 0.0 and self._desired_emotion[2] > 0.0):
            # Usually, high Dominance have the character arch back to elevate the shoulders. For angry (i.e when the pleasure is also low) this is the opposite
                if(coefficient < 1.1 and coefficient > 0.9):
                    change_weight_factor_x = 1.5
                    change_weight_factor_y = 1.5
                else:
                    change_weight_factor_x = 0.12
                    change_weight_factor_y = 0.12

                new_neck_position_x += 1.0 * ((coefficient - 1.0) * change_weight_factor_x) 
                new_neck_position_y -= 1.0 * ((coefficient - 1.0) * change_weight_factor_y) 
            else:
                if(self._desired_emotion[1] > 0.0 and self._desired_emotion[2] < 0.0):
                    if(coefficient > 1.0):
                        change_weight_factor_x = 0.025
                        change_weight_factor_y = 0.025
                    else:
                        change_weight_factor_x = 0.25
                        change_weight_factor_y = 0.25

                else:
                    if(coefficient > 1.0):
                        change_weight_factor_x = 0.025
                        change_weight_factor_y = 0.025
                    else:
                        change_weight_factor_x = 0.1
                        change_weight_factor_y = 0.1

                new_neck_position_x -= 1.0 * ((coefficient - 1.0) * change_weight_factor_x) 
                new_neck_position_y += 1.0 * ((coefficient - 1.0) * change_weight_factor_y) 

            gen_index = i

            print(self.generated_mocap[gen_index]['mocap']['neck'])
            self.generated_mocap[gen_index]['mocap']['neck'] = (
                new_neck_position_x, new_neck_position_y, new_neck_position_z)
            print(self.generated_mocap[gen_index]['mocap']['neck'])
            print()

    def rule_3(self):
        print("\n== RULE 3 - HANDS ==")
        # Move on Vector going from root to wrist and from wrist to neck

        coefficient_index = 0
        for i in range(len(self._mocap)):
            #1 C3 HANDS HIPS
            #2 C3 HANDS CHEST

            coefficient_hips = self.c3_1#[coefficient_index]
            coefficient_chest = self.c3_2#[coefficient_index]
            coefficient_index += 1

            current_root_position = self._mocap[i]["root"][0]
            current_head_position = self._mocap[i]["neck"][0]

            current_left_hand = self._mocap[i]["left_wrist"][0]
            current_right_hand = self._mocap[i]["right_wrist"][0]

            current_left_elbow = self._mocap[i]["left_elbow"][0]
            current_right_elbow = self._mocap[i]["right_elbow"][0]

            # Unit Vectors ROOT #
            # Left to Root #
            d_left_hips = np.asarray([current_left_hand[0] - current_root_position[0], 
                      current_left_hand[1] - current_root_position[1], 
                      current_left_hand[2] - current_root_position[2]])
            d_left_hips = d_left_hips / np.linalg.norm(d_left_hips)

            # Right to Root #
            d_right_hips = np.asarray([current_right_hand[0] - current_root_position[0], 
                      current_right_hand[1] - current_root_position[1], 
                      current_right_hand[2] - current_root_position[2]])
            d_right_hips = d_right_hips / np.linalg.norm(d_right_hips)

            new_left_hand_position_x = current_left_hand[0]
            new_left_hand_position_y = current_left_hand[1]
            new_left_hand_position_z = current_left_hand[2]

            new_right_hand_position_x = current_right_hand[0]
            new_right_hand_position_y = current_right_hand[1]
            new_right_hand_position_z = current_right_hand[2]

            # LEFT #
            new_left_hand_position_x += d_left_hips[0] * (coefficient_hips-1.0) * 0.5
            new_left_hand_position_y += d_left_hips[1] * (coefficient_hips-1.0) * 0.5
            new_left_hand_position_z += d_left_hips[2] * (coefficient_hips-1.0) * 0.5

            ## RIGHT #
            new_right_hand_position_x += d_right_hips[0] * (coefficient_hips-1.0) * 0.5
            new_right_hand_position_y += d_right_hips[1] * (coefficient_hips-1.0) * 0.5
            new_right_hand_position_z += d_right_hips[2] * (coefficient_hips-1.0) * 0.5


            # Unit Vectors HEAD #
            # Left to Head #
            d_left_head = np.asarray([current_head_position[0] + 0.4 - new_left_hand_position_x, 
                      current_head_position[1] - new_left_hand_position_y, 
                      current_head_position[2] - new_left_hand_position_z])
            d_left_head = d_left_head / np.linalg.norm(d_left_head)

            # Right to Head #
            d_right_head = np.asarray([current_head_position[0] + 0.4 - new_right_hand_position_x, 
                      current_head_position[1] - new_right_hand_position_y, 
                      current_head_position[2] - new_right_hand_position_z])
            d_right_head = d_right_head / np.linalg.norm(d_right_head)


            if(self._desired_emotion[1] < 0.0 and self._desired_emotion[2] < 0.0):
                # LEFT #
                new_left_hand_position_x -= d_left_head[0] * (coefficient_chest-1.0) * 0.5
                new_left_hand_position_y += d_left_head[1] * (coefficient_chest-1.0) * 0.5
                new_left_hand_position_z -= d_left_head[2] * (coefficient_chest-1.0) * 0.5

                # RIGHT #
                new_right_hand_position_x -= d_right_head[0] * (coefficient_chest-1.0) * 0.5
                new_right_hand_position_y += d_right_head[1] * (coefficient_chest-1.0) * 0.5
                new_right_hand_position_z -= d_right_head[2] * (coefficient_chest-1.0) * 0.5

                # Left to Elbow #
                #d_left_elbow = np.asarray([current_left_elbow[0] - new_left_hand_position_x, 
                #        0.0, 
                #        current_left_elbow[2] - new_left_hand_position_z])
                #d_left_elbow = d_left_elbow / np.linalg.norm(d_left_elbow)

                #new_left_hand_position_x -= d_left_elbow[0] * (coefficient_chest-1.0) * 0.2
                #new_left_hand_position_z -= d_left_elbow[2] * (coefficient_chest-1.0) * 0.2

                # Right to Elbow #
                #d_right_elbow = np.asarray([current_right_elbow[0] - new_right_hand_position_x, 
                #        0.0, 
                #        current_right_elbow[2] - new_right_hand_position_z])
                #d_right_elbow = d_right_elbow / np.linalg.norm(d_right_elbow)

                #new_right_hand_position_x -= d_right_elbow[0] * (coefficient_chest-1.0) * 0.2
                #new_right_hand_position_z -= d_right_elbow[2] * (coefficient_chest-1.0) * 0.2

            else:
                # LEFT #
                new_left_hand_position_x -= d_left_head[0] * (coefficient_chest-1.0) * 0.5
                new_left_hand_position_y -= d_left_head[1] * (coefficient_chest-1.0) * 0.5
                new_left_hand_position_z -= d_left_head[2] * (coefficient_chest-1.0) * 0.5

                # RIGHT #
                new_right_hand_position_x -= d_right_head[0] * (coefficient_chest-1.0) * 0.5
                new_right_hand_position_y -= d_right_head[1] * (coefficient_chest-1.0) * 0.5
                new_right_hand_position_z -= d_right_head[2] * (coefficient_chest-1.0) * 0.5

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
        # Move on vector going from root to elbow

        coefficient_index = 0
        for i in range(len(self._mocap)):
            coefficient = self.c4#[coefficient_index]
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

            #if((self._desired_emotion[0] < -0.3 and self._desired_emotion[1] > 0.5 and self._desired_emotion[2] < -0.3)): 
            #    #Usually, high Arousal emotions have broad elbows, but for afraid that is different
#
            #    new_left_elbow_pos_x -= d_left[0] * (coefficient-1.0) * 0.4
            #    new_left_elbow_pos_y -= d_left[1] * (coefficient-1.0) * 0.4
            #    new_left_elbow_pos_z -= d_left[2] * (coefficient-1.0) * 0.4
#
            #    new_right_elbow_pos_x -= d_right[0] * (coefficient-1.0) * 0.4
            #    new_right_elbow_pos_y -= d_right[1] * (coefficient-1.0) * 0.4
            #    new_right_elbow_pos_z -= d_right[2] * (coefficient-1.0) * 0.4
#
            #elif(self._desired_emotion[0] < 0.0 and self._desired_emotion[1] < 0.0 and self._desired_emotion[2] < 0.0):
            #    new_left_elbow_pos_x -= d_left[0] * (coefficient-1.0) * 0.3
            #    new_left_elbow_pos_y -= d_left[1] * (coefficient-1.0) * 0.3
            #    new_left_elbow_pos_z -= d_left[2] * (coefficient-1.0) * 0.3
#
            #    new_right_elbow_pos_x -= d_right[0] * (coefficient-1.0) * 0.3
            #    new_right_elbow_pos_y -= d_right[1] * (coefficient-1.0) * 0.3
            #    new_right_elbow_pos_z -= d_right[2] * (coefficient-1.0) * 0.3
#
            #else:
            #    new_left_elbow_pos_x += d_left[0] * (coefficient-1.0) * 0.5
            #    new_left_elbow_pos_y += d_left[1] * (coefficient-1.0) * 0.4
            #    new_left_elbow_pos_z += d_left[2] * (coefficient-1.0) * 0.4
#
            #    new_right_elbow_pos_x += d_right[0] * (coefficient-1.0) * 0.5
            #    new_right_elbow_pos_y += d_right[1] * (coefficient-1.0) * 0.4
            #    new_right_elbow_pos_z += d_right[2] * (coefficient-1.0) * 0.4

            new_left_elbow_pos_x += d_left[0] * (coefficient-1.0) * 0.5
            new_left_elbow_pos_y += d_left[1] * (coefficient-1.0) * 0.5
            new_left_elbow_pos_z += d_left[2] * (coefficient-1.0) * 0.5

            new_right_elbow_pos_x += d_right[0] * (coefficient-1.0) * 0.5
            new_right_elbow_pos_y += d_right[1] * (coefficient-1.0) * 0.5
            new_right_elbow_pos_z += d_right[2] * (coefficient-1.0) * 0.5

            if(abs(new_left_elbow_pos_z - current_root_position[2]) < 0.15):
                new_left_elbow_pos_z = current_root_position[2] - 0.15

            if(abs(new_right_elbow_pos_z - current_root_position[2]) < 0.15):
                new_right_elbow_pos_z = current_root_position[2] + 0.15

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

    def rule_5(self):
        print("\n== RULE 5 - FEET ==")
        # Move on Vector going from left foot to right and right foot to left (ignoring height)

        coefficient_index = 0
        for i in range(len(self._mocap)):
            coefficient = self.c5#[coefficient_index]
            coefficient_index += 1

            coefficient = min(coefficient, 1.5) # We dont want this coefficient to be too large
            coefficient = max(coefficient, 0.5) # Or too small

            current_left_ankle_pos = self._mocap[i]["left_ankle"][0]
            current_right_ankle_pos = self._mocap[i]["right_ankle"][0]
            current_root_pos = self._mocap[i]["root"][0]

            # Unit Vectors #
            d_right_2_left = np.asarray([current_left_ankle_pos[0] - current_right_ankle_pos[0], 
                      0.0, 
                      current_left_ankle_pos[2] - current_right_ankle_pos[2]])
            d_right_2_left = d_right_2_left / np.linalg.norm(d_right_2_left)
        
            d_left_2_right = np.asarray([current_right_ankle_pos[0] - current_left_ankle_pos[0], 
                      0.0, 
                      current_right_ankle_pos[2] - current_left_ankle_pos[2]])
            d_left_2_right = d_left_2_right / np.linalg.norm(d_left_2_right)


            # Left #
            new_left_ankle_pos_x = current_left_ankle_pos[0]
            new_left_ankle_pos_y = current_left_ankle_pos[1]
            new_left_ankle_pos_z = current_left_ankle_pos[2]

            # Right #
            new_right_ankle_pos_x = current_right_ankle_pos[0]
            new_right_ankle_pos_y = current_right_ankle_pos[1]
            new_right_ankle_pos_z = current_right_ankle_pos[2]

            new_left_ankle_pos_x -= d_left_2_right[0] * (coefficient-1.0) * 0.2
            new_left_ankle_pos_z -= d_left_2_right[2] * (coefficient-1.0) * 0.2

            new_right_ankle_pos_x -= d_right_2_left[0] * (coefficient-1.0) * 0.2
            new_right_ankle_pos_z -= d_right_2_left[2] * (coefficient-1.0) * 0.2



            gen_index = i

            print(self.generated_mocap[gen_index]
                  ['mocap']['left_ankle'])
            self.generated_mocap[gen_index]['mocap']['left_ankle'] = (
                new_left_ankle_pos_x, new_left_ankle_pos_y, new_left_ankle_pos_z)
            print(self.generated_mocap[gen_index]
                  ['mocap']['left_ankle'])
            print()         

            print(self.generated_mocap[gen_index]
                  ['mocap']['right_ankle'])
            self.generated_mocap[gen_index]['mocap']['right_ankle'] = (
                new_right_ankle_pos_x, new_right_ankle_pos_y, new_right_ankle_pos_z)
            print(self.generated_mocap[gen_index]
                  ['mocap']['right_ankle'])
            print()
            

    def rule_6(self):
        #print("\n== RULE 6 - NECK ROTATION ==")
        # Rotate head on the Z axis to make it look down or up

        coefficient_index = 0
        for i in range(len(self._mocap)):
            coefficient = self.c6
            coefficient_index += 1

            pose = self._mocap[i]["frame"]
            current_neck_rotation = p1.getEulerFromQuaternion([pose[12],pose[13],pose[14],pose[11]])
            
            new_neck_rotation_x = current_neck_rotation[0]
            new_neck_rotation_y = current_neck_rotation[1]
            new_neck_rotation_z = current_neck_rotation[2]

            # Pos Angle -> head looks up ; Neg Angle -> head looks down
            new_neck_rotation_z = (coefficient - 1.0) * 1.5 + (self._desired_emotion[0] * 0.5)
            
            if(self._desired_emotion[1] > 0.0 and new_neck_rotation_z < -0.1):
                # Don't let high arousal emotions (like fear or anger) have a slumped neck
                new_neck_rotation_z = -0.1

            # Clamp neck rotation to avoid broken necks
            new_neck_rotation_z = max(new_neck_rotation_z, -math.pi/3)
            new_neck_rotation_z = min(new_neck_rotation_z, math.pi/12)

            desired_rotation = [new_neck_rotation_x, new_neck_rotation_y, new_neck_rotation_z]

            gen_index = i

            print(current_neck_rotation[2])
            self.generated_mocap[gen_index]['orn']['neck'] = (
                desired_rotation[0], desired_rotation[1], desired_rotation[2])
            print(new_neck_rotation_z)
            print()


    def rule_1_single(self, frame, coefficient):
        #print("\n== RULE 1 SINGLE - PELVIS ==")

        coefficient = self.c1
            
        current_root_height = frame["root"][1]
        
        new_root_height = current_root_height

        new_root_height += 1.0 * ((coefficient - 1.0) * 0.08) #0.1 -> dampening factor

        generated_pose = (
                frame["root"][0], new_root_height, frame["root"][2])

        return generated_pose

    def rule_2_single(self, frame, coefficient):
        #print("\n== RULE 2 SINGLE - NECK ==")

        t_pose_root = T_POSE["root"][0]
        t_pose_head = T_POSE["neck"][0]
        
        # Vector from Head to Pelvis in T-Pose
        dt_head = (t_pose_root[0] - t_pose_head[0], 
                   t_pose_root[1] - t_pose_head[1],
                   t_pose_root[2] - t_pose_head[2])

        coefficient = self.c2

        current_root_position = frame["root"]
        current_neck_position = frame["neck"]

        # Vector from Head to Pelvis in Current Pose
        d_head = (current_root_position[0] - current_neck_position[0], 
                  current_root_position[1] - current_neck_position[1], 
                  current_root_position[2] - current_neck_position[2])

        new_neck_position_x = current_neck_position[0]
        new_neck_position_y = current_neck_position[1]
        new_neck_position_z = current_neck_position[2]

            
        if(self._desired_emotion[0] < 0.0 and self._desired_emotion[2] > 0.0):
        # Usually, high Dominance have the character arch back to elevate the shoulders. For angry (i.e when the pleasure is also low) this is the opposite
            if(coefficient < 1.1 and coefficient > 0.9):
                change_weight_factor_x = 1.5
                change_weight_factor_y = 1.5
            else:
                change_weight_factor_x = 0.12
                change_weight_factor_y = 0.12

            new_neck_position_x += 1.0 * ((coefficient - 1.0) * change_weight_factor_x) 
            new_neck_position_y -= 1.0 * ((coefficient - 1.0) * change_weight_factor_y) 
        else:
            if(self._desired_emotion[1] > 0.0 and self._desired_emotion[2] < 0.0):
                if(coefficient > 1.0):
                    change_weight_factor_x = 0.025
                    change_weight_factor_y = 0.025
                else:
                    change_weight_factor_x = 0.25
                    change_weight_factor_y = 0.25
            else:
                if(coefficient > 1.0):
                    change_weight_factor_x = 0.025
                    change_weight_factor_y = 0.025
                else:
                    change_weight_factor_x = 0.1
                    change_weight_factor_y = 0.1

            new_neck_position_x -= 1.0 * ((coefficient - 1.0) * change_weight_factor_x) 
            new_neck_position_y += 1.0 * ((coefficient - 1.0) * change_weight_factor_y)

        generated_pose = (
                new_neck_position_x, new_neck_position_y, new_neck_position_z)

        #generated_pose = (frame["neck"][0], frame["neck"][1], frame["neck"][2])

        return generated_pose

    def rule_3_single(self, frame, coefficient_1, coefficient_2):
        #print("\n== RULE 3 SINGLE - HANDS ==")
        # Move on Vector going from root to wrist and from wrist to neck

        #1 C3 HANDS HIPS
        #2 C3 HANDS CHEST

        coefficient_hips = self.c3_1#[coefficient_index]
        coefficient_chest = self.c3_2#[coefficient_index]

        current_root_position = frame["root"]
        current_head_position = frame["neck"]

        current_left_hand = frame["left_wrist"]
        current_right_hand = frame["right_wrist"]

        current_left_elbow = frame["left_elbow"]
        current_right_elbow = frame["right_elbow"]

        # Unit Vectors ROOT #
        # Left to Root #
        d_left_hips = np.asarray([current_left_hand[0] - current_root_position[0], 
                  current_left_hand[1] - current_root_position[1], 
                  current_left_hand[2] - current_root_position[2]])
        d_left_hips = d_left_hips / np.linalg.norm(d_left_hips)

        # Right to Root #
        d_right_hips = np.asarray([current_right_hand[0] - current_root_position[0], 
                  current_right_hand[1] - current_root_position[1], 
                  current_right_hand[2] - current_root_position[2]])
        d_right_hips = d_right_hips / np.linalg.norm(d_right_hips)

        new_left_hand_position_x = current_left_hand[0]
        new_left_hand_position_y = current_left_hand[1]
        new_left_hand_position_z = current_left_hand[2]

        new_right_hand_position_x = current_right_hand[0]
        new_right_hand_position_y = current_right_hand[1]
        new_right_hand_position_z = current_right_hand[2]

        # LEFT #
        new_left_hand_position_x += d_left_hips[0] * (coefficient_hips-1.0) * 0.5
        new_left_hand_position_y += d_left_hips[1] * (coefficient_hips-1.0) * 0.5
        new_left_hand_position_z += d_left_hips[2] * (coefficient_hips-1.0) * 0.5

        ## RIGHT #
        new_right_hand_position_x += d_right_hips[0] * (coefficient_hips-1.0) * 0.5
        new_right_hand_position_y += d_right_hips[1] * (coefficient_hips-1.0) * 0.5
        new_right_hand_position_z += d_right_hips[2] * (coefficient_hips-1.0) * 0.5


        # Unit Vectors HEAD #
        # Left to Head #
        d_left_head = np.asarray([current_head_position[0] + 0.4 - new_left_hand_position_x, 
                  current_head_position[1] - new_left_hand_position_y, 
                  current_head_position[2] - new_left_hand_position_z])
        d_left_head = d_left_head / np.linalg.norm(d_left_head)

        # Right to Head #
        d_right_head = np.asarray([current_head_position[0] + 0.4 - new_right_hand_position_x, 
                  current_head_position[1] - new_right_hand_position_y, 
                  current_head_position[2] - new_right_hand_position_z])
        d_right_head = d_right_head / np.linalg.norm(d_right_head)


        if(self._desired_emotion[1] < 0.0 and self._desired_emotion[2] < 0.0):
            # LEFT #
            new_left_hand_position_x -= d_left_head[0] * (coefficient_chest-1.0) * 0.5
            new_left_hand_position_y += d_left_head[1] * (coefficient_chest-1.0) * 0.5
            new_left_hand_position_z -= d_left_head[2] * (coefficient_chest-1.0) * 0.5

            # RIGHT #
            new_right_hand_position_x -= d_right_head[0] * (coefficient_chest-1.0) * 0.5
            new_right_hand_position_y += d_right_head[1] * (coefficient_chest-1.0) * 0.5
            new_right_hand_position_z -= d_right_head[2] * (coefficient_chest-1.0) * 0.5

            # Left to Elbow #
            #d_left_elbow = np.asarray([current_left_elbow[0] - new_left_hand_position_x, 
            #        0.0, 
            #        current_left_elbow[2] - new_left_hand_position_z])
            #d_left_elbow = d_left_elbow / np.linalg.norm(d_left_elbow)
            #new_left_hand_position_x -= d_left_elbow[0] * (coefficient_chest-1.0) * 0.2
            #new_left_hand_position_z -= d_left_elbow[2] * (coefficient_chest-1.0) * 0.2

            # Right to Elbow #
            #d_right_elbow = np.asarray([current_right_elbow[0] - new_right_hand_position_x, 
            #        0.0, 
            #        current_right_elbow[2] - new_right_hand_position_z])
            #d_right_elbow = d_right_elbow / np.linalg.norm(d_right_elbow)
            #new_right_hand_position_x -= d_right_elbow[0] * (coefficient_chest-1.0) * 0.2
            #new_right_hand_position_z -= d_right_elbow[2] * (coefficient_chest-1.0) * 0.2

        else:
            # LEFT #
            new_left_hand_position_x -= d_left_head[0] * (coefficient_chest-1.0) * 0.5
            new_left_hand_position_y -= d_left_head[1] * (coefficient_chest-1.0) * 0.5
            new_left_hand_position_z -= d_left_head[2] * (coefficient_chest-1.0) * 0.5

            # RIGHT #
            new_right_hand_position_x -= d_right_head[0] * (coefficient_chest-1.0) * 0.5
            new_right_hand_position_y -= d_right_head[1] * (coefficient_chest-1.0) * 0.5
            new_right_hand_position_z -= d_right_head[2] * (coefficient_chest-1.0) * 0.5


        generated_pose_l = (
                new_left_hand_position_x, new_left_hand_position_y, new_left_hand_position_z)

        generated_pose_r = (
                new_right_hand_position_x, new_right_hand_position_y, new_right_hand_position_z)

        #generated_pose_l = (frame["left_wrist"][0], frame["left_wrist"][1], frame["left_wrist"][2])
        #generated_pose_r = (frame["right_wrist"][0], frame["right_wrist"][1], frame["right_wrist"][2])

        return (generated_pose_l, generated_pose_r)

    def rule_4_single(self, frame, coefficient):
        #print("\n== RULE 4 SINGLE - ELBOWS ==")

        coefficient = self.c4#[coefficient_index]

        current_left_elbow_pos = frame["left_elbow"]
        current_right_elbow_pos = frame["right_elbow"]
        current_root_position = frame["root"]

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

        new_left_elbow_pos_x += d_left[0] * (coefficient-1.0) * 0.5
        new_left_elbow_pos_y += d_left[1] * (coefficient-1.0) * 0.5
        new_left_elbow_pos_z += d_left[2] * (coefficient-1.0) * 0.5

        new_right_elbow_pos_x += d_right[0] * (coefficient-1.0) * 0.5
        new_right_elbow_pos_y += d_right[1] * (coefficient-1.0) * 0.5
        new_right_elbow_pos_z += d_right[2] * (coefficient-1.0) * 0.5

        if(abs(new_left_elbow_pos_z - current_root_position[2]) < 0.15):
            new_left_elbow_pos_z = current_root_position[2] - 0.15

        if(abs(new_right_elbow_pos_z - current_root_position[2]) < 0.15):
            new_right_elbow_pos_z = current_root_position[2] + 0.15

        generated_pose_l = (
                new_left_elbow_pos_x, new_left_elbow_pos_y, new_left_elbow_pos_z)

        generated_pose_r = (
                new_right_elbow_pos_x, new_right_elbow_pos_y, new_right_elbow_pos_z)

        #generated_pose_l = (frame["left_elbow"][0], frame["left_elbow"][1], frame["left_elbow"][2])
        #generated_pose_r = (frame["right_elbow"][0], frame["right_elbow"][1], frame["right_elbow"][2])

        return (generated_pose_l, generated_pose_r)

    def rule_5_single(self, frame, coefficient):
        #print("\n== RULE 5 SINGLE - FEET ==")
        # Move on Vector going from left foot to right and right foot to left (ignoring height)

        coefficient = self.c5#[coefficient_index]

        coefficient = min(coefficient, 1.5) # We dont want this coefficient to be too large
        coefficient = max(coefficient, 0.5) # Or too small

        current_left_ankle_pos = frame["left_ankle"]
        current_right_ankle_pos = frame["right_ankle"]
        current_root_pos = frame["root"]

        # Unit Vectors #
        d_right_2_left = np.asarray([current_left_ankle_pos[0] - current_right_ankle_pos[0], 
                  0.0, 
                  current_left_ankle_pos[2] - current_right_ankle_pos[2]])
        d_right_2_left = d_right_2_left / np.linalg.norm(d_right_2_left)
    
        d_left_2_right = np.asarray([current_right_ankle_pos[0] - current_left_ankle_pos[0], 
                  0.0, 
                  current_right_ankle_pos[2] - current_left_ankle_pos[2]])
        d_left_2_right = d_left_2_right / np.linalg.norm(d_left_2_right)


        # Left #
        new_left_ankle_pos_x = current_left_ankle_pos[0]
        new_left_ankle_pos_y = current_left_ankle_pos[1]
        new_left_ankle_pos_z = current_left_ankle_pos[2]

        # Right #
        new_right_ankle_pos_x = current_right_ankle_pos[0]
        new_right_ankle_pos_y = current_right_ankle_pos[1]
        new_right_ankle_pos_z = current_right_ankle_pos[2]

        new_left_ankle_pos_x -= d_left_2_right[0] * (coefficient-1.0) * 0.2
        new_left_ankle_pos_z -= d_left_2_right[2] * (coefficient-1.0) * 0.2

        new_right_ankle_pos_x -= d_right_2_left[0] * (coefficient-1.0) * 0.2
        new_right_ankle_pos_z -= d_right_2_left[2] * (coefficient-1.0) * 0.2


        generated_pose_l = (
                new_left_ankle_pos_x, new_left_ankle_pos_y, new_left_ankle_pos_z)

        generated_pose_r = (
                new_right_ankle_pos_x, new_right_ankle_pos_y, new_right_ankle_pos_z)

        #generated_pose_l = (frame["left_ankle"][0], frame["left_ankle"][1], frame["left_ankle"][2])
        #generated_pose_r = (frame["right_ankle"][0], frame["right_ankle"][1], frame["right_ankle"][2])

        return (generated_pose_l, generated_pose_r)

    def rule_6_single(self, frame, coefficient):
        #print("\n== RULE 6 SINGLE - NECK ROTATION ==")

        coefficient = self.c6

        pose = frame
        current_neck_rotation = p1.getEulerFromQuaternion([pose[12],pose[13],pose[14],pose[11]])
            
        new_neck_rotation_x = current_neck_rotation[0]
        new_neck_rotation_y = current_neck_rotation[1]
        new_neck_rotation_z = current_neck_rotation[2]

            
        # Pos Angle -> head looks up ; Neg Angle -> head looks down
        new_neck_rotation_z = (coefficient - 1.0) * 1.5 + (self._desired_emotion[0] * 0.5)
            
        if(self._desired_emotion[1] > 0.0 and new_neck_rotation_z < -0.1):
            # Don't let high arousal emotions (like fear or anger) have a slumped neck
            new_neck_rotation_z = -0.1

        # Clamp neck rotation to avoid broken necks
        new_neck_rotation_z = max(new_neck_rotation_z, -math.pi/3)
        new_neck_rotation_z = min(new_neck_rotation_z, math.pi/12)

        desired_rotation = [new_neck_rotation_x, new_neck_rotation_y, new_neck_rotation_z]

        return desired_rotation

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
