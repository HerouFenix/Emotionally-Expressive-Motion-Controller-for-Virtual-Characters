import math
import numpy as np
import os.path

import pandas as pd

import xgboost as xgb

import joblib


xgb.set_config(verbosity=0)

class EmotionClassifier():
    def __init__(self):
        self._model_p = xgb.XGBRegressor(verbosity=0)
        self._model_p.load_model("../emotion_classifier/models/l2p_model_kin.json")

        self._model_a = xgb.XGBRegressor(verbosity=0)
        self._model_a.load_model("../emotion_classifier/models/l2a_model_kin.json")

        self._model_d = xgb.XGBRegressor(verbosity=0)
        self._model_d.load_model("../emotion_classifier/models/l2d_model_kin.json")

        self.p_predictions = []
        self.a_predictions = []
        self.d_predictions = []
        
        self.normalizer = joblib.load(r'../emotion_classifier/models/scalers/Fs_B_S_DANCE_WALK_KIN_0.5sec.pkl') 

        self.predicted_p = 0.0
        self.predicted_a = 0.0
        self.predicted_d = 0.0


    def predict_emotion_coordinates(self, lma_features):    
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


        self.p_predictions.append(self.predicted_p)
        self.a_predictions.append(self.predicted_a)
        self.d_predictions.append(self.predicted_d)

        print((self.predicted_p, self.predicted_a, self.predicted_d))

        return (self.predicted_p, self.predicted_a, self.predicted_d)

    def predict_final_emotion(self):
        
        # Get largest absolute values
        largest_p = max(self.p_predictions, key=abs)
        max_p_i = self.p_predictions.index(largest_p)

        largest_a = max(self.a_predictions, key=abs)
        max_a_i = self.a_predictions.index(largest_a)
        
        largest_d = max(self.d_predictions, key=abs)
        max_d_i = self.d_predictions.index(largest_d)

        # Get # pos and neg
        p_neg_count = len(list(filter(lambda x: (x < 0), self.p_predictions)))
        p_pos_count = len(list(filter(lambda x: (x >= 0), self.p_predictions)))

        a_neg_count = len(list(filter(lambda x: (x < 0), self.a_predictions)))
        a_pos_count = len(list(filter(lambda x: (x >= 0), self.a_predictions)))

        d_neg_count = len(list(filter(lambda x: (x < 0), self.d_predictions)))
        d_pos_count = len(list(filter(lambda x: (x >= 0), self.d_predictions)))


        if((p_pos_count > p_neg_count and largest_p > 0) or (p_neg_count > p_pos_count and largest_p < 0)):
            self.p_predictions.pop(max_p_i)
            self.predicted_p = (sum(self.p_predictions)/len(self.p_predictions)) * 0.2 + largest_p * 0.8
        else:
            self.predicted_p = sum(self.p_predictions)/len(self.p_predictions)

        if((a_pos_count > a_neg_count and largest_a > 0) or (a_neg_count > a_pos_count and largest_a < 0)):
            self.a_predictions.pop(max_a_i)
            self.predicted_a = (sum(self.a_predictions)/len(self.a_predictions)) * 0.2 + largest_a * 0.8
        else:
            self.predicted_a = sum(self.a_predictions)/len(self.a_predictions)

        if((d_pos_count > d_neg_count and largest_d > 0) or (d_neg_count > d_pos_count and largest_d < 0)):
            self.d_predictions.pop(max_d_i)
            self.predicted_d = (sum(self.d_predictions)/len(self.d_predictions)) * 0.2 + largest_d * 0.8
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