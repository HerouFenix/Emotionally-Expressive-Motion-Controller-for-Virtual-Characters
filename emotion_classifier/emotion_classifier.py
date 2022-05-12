import math
import numpy as np
import os.path

import pandas as pd

import xgboost as xgb

import joblib

from gui_manager import GUIManager

xgb.set_config(verbosity=0)

class EmotionClassifier():
    def __init__(self):
        self._model_p = xgb.XGBRegressor(verbosity=0)
        self._model_p.load_model("../emotion_classifier/models/l2p_dance_model_kin.json")

        self._model_a = xgb.XGBRegressor(verbosity=0)
        self._model_a.load_model("../emotion_classifier/models/l2a_dance_model_kin.json")

        self._model_d = xgb.XGBRegressor(verbosity=0)
        self._model_d.load_model("../emotion_classifier/models/l2d_dance_model_kin.json")

        self.p_predictions = []
        self.a_predictions = []
        self.d_predictions = []
        
        self.normalizer = joblib.load(r'../emotion_classifier/models/scalers/Fs_B_O_S_DANCE_WALK_KIN_0.5sec.pkl') 

        self.predicted_p = 0.0
        self.predicted_a = 0.0
        self.predicted_d = 0.0


    def predict_emotion_coordinates(self, lma_features, results_store = None):    
        """
        Gets array of lma features and predicts each of their pleasure and arousal
        """
          
                
        rows = []
        
        for entry in lma_features:
            row = []
            for feature in entry["lma_features"]:
                if(type(feature) is tuple):
                    for i in feature:
                        row.append(i)
                else:
                    row.append(feature)

            rows.append(row)


        # Normalize data
        df = pd.DataFrame(rows, columns=[
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
         ])
        
        df = self.normalizer.transform(df)


        y_p = self._model_p.predict(df)
        y_a = self._model_a.predict(df)
        y_d = self._model_d.predict(df)


        self.predicted_p = sum(y_p)/len(y_p)
        self.predicted_a = sum(y_a)/len(y_a)
        self.predicted_d = sum(y_d)/len(y_d)


        #self.p_predictions.append(self.predicted_p)
        #self.a_predictions.append(self.predicted_a)
        #self.d_predictions.append(self.predicted_d)
        self.p_predictions += y_p.tolist()
        self.a_predictions += y_a.tolist()
        self.d_predictions += y_d.tolist()

        for i in range(len(y_p)):
            print((y_p[i], y_a[i], y_d[i]))

        if(results_store is not None):
            results_store[0] = self.predicted_p
            results_store[1] = self.predicted_a
            results_store[2] = self.predicted_d

        return (self.predicted_p, self.predicted_a, self.predicted_d)

    def predict_final_emotion(self):
        # Sort Lists according to max absolute value
        self.p_predictions = sorted(self.p_predictions, key=abs, reverse=True)
        self.a_predictions = sorted(self.a_predictions, key=abs, reverse=True)
        self.d_predictions = sorted(self.d_predictions, key=abs, reverse=True)


        # Get largest absolute values
        #largest_p = max(self.p_predictions, key=abs)
        #max_p_i = self.p_predictions.index(largest_p)

        #largest_a = max(self.a_predictions, key=abs)
        #max_a_i = self.a_predictions.index(largest_a)
        
        #largest_d = max(self.d_predictions, key=abs)
        #max_d_i = self.d_predictions.index(largest_d)

        # Get # pos and neg
        p_neg_count = len(list(filter(lambda x: (x < 0), self.p_predictions)))
        p_pos_count = len(self.p_predictions) - p_neg_count

        a_neg_count = len(list(filter(lambda x: (x < 0), self.a_predictions)))
        a_pos_count = len(self.a_predictions) - a_neg_count

        d_neg_count = len(list(filter(lambda x: (x < 0), self.d_predictions)))
        d_pos_count = len(self.d_predictions) - d_neg_count


        largest_p = 0
        max_p_i = -1
        
        largest_a = 0
        max_a_i = -1

        largest_d = 0
        max_d_i = -1

        for i in range(3):
            if(max_p_i == -1 and ((p_pos_count > p_neg_count and self.p_predictions[i] > 0) or (p_neg_count > p_pos_count and self.p_predictions[i] < 0))):
                largest_p = self.p_predictions[i]
                max_p_i = i
                
            if(max_a_i == -1 and ((a_pos_count > a_neg_count and self.a_predictions[i] > 0) or (a_neg_count > a_pos_count and self.a_predictions[i] < 0))):
                largest_a = self.a_predictions[i]
                max_a_i = i
            
            if(max_d_i == -1 and ((d_pos_count > d_neg_count and self.d_predictions[i] > 0) or (d_neg_count > d_pos_count and self.d_predictions[i] < 0))):
                largest_d = self.d_predictions[i]
                max_d_i = i

        if(max_p_i != -1):
            self.p_predictions.pop(max_p_i)
            self.predicted_p = (sum(self.p_predictions)/len(self.p_predictions)) * 0.8 + largest_p * 0.2
        else:
            self.predicted_p = sum(self.p_predictions)/len(self.p_predictions)

        if(max_a_i != -1):
            self.a_predictions.pop(max_a_i)
            self.predicted_a = (sum(self.a_predictions)/len(self.a_predictions)) * 0.8 + largest_a * 0.2
        else:
            self.predicted_a = sum(self.a_predictions)/len(self.a_predictions)

        if(max_d_i != -1):
            self.d_predictions.pop(max_d_i)
            self.predicted_d = (sum(self.d_predictions)/len(self.d_predictions)) * 0.8 + largest_d * 0.2
        else:
            self.predicted_d = sum(self.d_predictions)/len(self.d_predictions)


        print("==Final Prediction==\n",(self.predicted_p, self.predicted_a, self.predicted_d))
        return (self.predicted_p, self.predicted_a, self.predicted_d)

    def clear(self):
        self.p_predictions = []
        self.a_predictions = []
        self.d_predictions = []

        self.predicted_p = 0.0
        self.predicted_a = 0.0
        self.predicted_d = 0.0