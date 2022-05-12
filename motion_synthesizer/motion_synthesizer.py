import math
import numpy as np
import os.path

import pandas as pd
from ast import literal_eval

from scipy.optimize import minimize

#from gui_manager import GUIManager

ANGRY_LMA = [ 0.04311089, -0.17033504, -0.40539778,  0.40837655,  0.76885704,  0.6150567,
   0.65320629,  0.4599897,  -0.19115373,  0.35164264,  0.25092236,  0.01501756,
   0.10871071,  0.51482174, -0.76377985,  1.25444347, -0.88972886, -0.33560446,
  -0.75217765,  0.84085059, -0.55450077,  0.82887554, -0.61220656, -0.90267698,
   0.36426269, -0.7325849,   0.49418937]

SAD_LMA = [-0.95938538, -0.54427577,  0.21936451, -1.35390726, -1.5422353,  -2.34866785,
  -1.25344367, -2.18366987, -0.25012049, -0.85340873,  0.68212738, -0.2148421,
  -0.31937091,  0.38375279, -0.42554333, -1.6237051, -1.21546882, -0.91020268,
  -0.90272286, -0.46156571, -0.45308077, -0.83833749, -0.97130512, -1.0795376,
  -0.9204676,  -0.89469554, -0.97541594]

C1_INDICES = [8, 14, 16]
C2_INDICES = [9, 10,11,12,13, 14,15]
C3_INDICES = [0, 1, 2, 4, 5, 6, 7, 14, 15, ]
C4_INDICES = [3, 17,18,19,20,21,22,23,24,25,26]

class MotionSynthesizer():
    def __init__(self, mocap_file, lma_file, reference_lma):

        self.c1 = 1.0
        self.c2 = 1.0
        self.c3 = 1.0
        self.c4 = 1.0

        self.current_reference_features = []
        self.current_features = []

        #1- Get MOCAP File
        
        self._mocap = []
        with open(mocap_file, 'r') as r:
            lines = r.readlines()

            for line in lines:
                line = literal_eval(line)
                del line["frame"]
                self._mocap.append(line)
        """
        #1- Get MOCAP File
        with open(mocap_file, 'r') as r:
            lines = r.readlines()

            mocap_str = ""
            for line in lines:
                if(line[0] != "[" and line[0] != "]"):
                    continue
                
                mocap_str += line
            
            self._mocap = literal_eval(mocap_str)      
        """
        
        #2 - Get MOCAP LMA File
        self._lma = []
        with open(lma_file, 'r') as r:
            lines = r.readlines()

            for line in lines:
                line = literal_eval(line)
                features = []
                for feature in line["lma_features"]:
                    if(type(feature) is tuple):
                        for i in range(len(feature)):
                            features.append(feature[i])
                    else:
                        features.append(feature)

                line["lma_features"] = features
                self._lma.append(line)


            self._keyframe = self._lma[0]["frame_counter"]
 
        #3 - Get REFERENCE LMA
        self._reference = reference_lma

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

        coefficient = 1.0

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

        res = minimize(self.compute_sse,
                  coefficient,
                  method   = 'Nelder-Mead',
                  #callback = self.minimize_callback,
                  options  = {'maxiter':50000,'disp': False})

        coefficient = res.x[0]
        print(coefficient)

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
            norms.append(np.linalg.norm(reference - (current_features[i] * coefficient)) ** 2)

        return norms

    def compute_sse(self, coefficient):
        norms = self.compute_norms(self.reference_features, self.current_features, coefficient)
        return np.sum(norms)

    def minimize_callback(self, coefficient):
        print("Coefficient: " + str(coefficient) + " | SSE: " + str(self.compute_sse(coefficient)))


#1 - GET MOCAP FILE [DONE]
#2 - GET MOCAP LMA FILE [DONE]
#3 - GET REFERENCE LMA [DONE]
#4 - SYNTHESIZE
#5a - STANDERDIZE NEW LMA FEATURES
#5b - EMOTION CLASSIFICATION TO SEE EMOTION OF NEW FEATURES

#Note that we apply motion synthesis only on sampled keyframes, i.e., every 5fth frame uniformly drawn from the motion
#We then interpolate the edited keyframes to obtain the full motion (NOTE: Pelo menos para o mocap do neutral walk aquilo esta de 5 em 5 keyframes)

# we 1st optimize for poses that satisfy the desired features as best as possible,  
# To this end, we designed four heuristic rules as shown in Table 2 to modify the positions of four control joints: head, hands, and pelvis

# Portanto temos 4 regras heuristicas, 1 para cada core joint (cabeca, l_hand r_hand e pelvis). Cada regra diz respeito a um set de LMA features diferente
# Cada uma destas regras depois usa um coeficiente para calcular a nova posicao destas core joints

# We search for c1, the coefficient for rule g1, that minimizes the distance between the current feature values and the desired feature values for all the
# keyframes indexed by time t in a least-squares sense (ou seja, encontramos o coeficiente que minimiza para TODAS as keyframes)
# To search for each coefficient we use interior-relective Newton method 

#  From these optimized coeffcients we then compute the desired positions for all the control joints on each keyframe.
#  We then pass these joint positions to an Inverse Kinematics solver (NOTE: Not sure se obrigatorio, try first without it, mas e capaz de ser preciso para encontrarmos a posiçao de cenas tipo elbows e shoulders given the hands)

ms = MotionSynthesizer("neutral_walk_mocap.txt", "neutral_walk_lma.txt", ANGRY_LMA)
ms.compute_coefficient(1)
ms.compute_coefficient(2)
ms.compute_coefficient(3)
ms.compute_coefficient(4)


# Coefficients
# c1 - Pelvis
# c2 - Head
# c3 - Hands
# c4 - FrameRate
# c5(?) - ELbows

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