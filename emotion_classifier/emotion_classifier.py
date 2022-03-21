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
        self._model_p.load_model("../emotion_classifier/l2p_model_final_v2.json")

        self._model_a = xgb.XGBRegressor(verbosity=0)
        self._model_a.load_model("../emotion_classifier/l2a_model_final_v2.json")

        self._model_d = xgb.XGBRegressor(verbosity=0)
        self._model_d.load_model("../emotion_classifier/l2d_model_final_v2.json")

        self.p_predictions = []
        self.a_predictions = []
        self.d_predictions = []
        
        self.normalizer = joblib.load(r'../emotion_classifier/scaler.pkl') 

        self.max_p = 0
        self.max_p_i = -1 # index
        self.max_a = 0
        self.max_a_i = -1 # index
        self.max_d = 0
        self.max_d_i = -1 # index

        self.p_pos_count = 0
        self.p_neg_count = 0

        self.a_pos_count = 0
        self.a_neg_count = 0

        self.d_pos_count = 0
        self.d_neg_count = 0

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

            # Preprocess row data - remove Total Body Volume
            del row[-2]

            rows.append(row)


        # Normalize data
        rows = self.normalizer.transform(rows)

        df = pd.DataFrame(rows, columns=['avg_hand_distance', 'avg_l_hand_hip_distance', 'avg_r_hand_hip_distance', 'avg_feet_distance', 'avg_l_hand_chest_distance', 'avg_r_hand_chest_distance', 'avg_l_elbow_hip_distance', 'avg_r_elbow_hip_distance', 'avg_chest_pelvis_distance', 'avg_neck_chest_distance', 'avg_neck_rotation_w', 'avg_neck_rotation_x', 'avg_neck_rotation_y', 'avg_neck_rotation_z',
         'avg_pelvis_rotation_w', 'avg_pelvis_rotation_x', 'avg_pelvis_rotation_y', 'avg_pelvis_rotation_z', 'std_l_hand_position', 'std_r_hand_position', 'avg_l_forearm_velocity', 
         'avg_r_forearm_velocity', 'avg_pelvis_velocity_x', 'avg_pelvis_velocity_y', 'avg_pelvis_velocity_z', 'avg_l_foot_velocity_x', 
         'avg_l_foot_velocity_y', 'avg_l_foot_velocity_z', 'avg_r_foot_velocity_x', 'avg_r_foot_velocity_y', 'avg_r_foot_velocity_z', 'avg_upper_body_volume', 
         'avg_distance_traveled'])

        y_p = self._model_p.predict(df)
        y_a = self._model_a.predict(df)
        y_d = self._model_d.predict(df)

        #for i in range(len(y_p)):
        #    self.p_predictions.append(y_p[i])
        #    self.a_predictions.append(y_a[i])
        #
        #    if(abs(y_p[i]) > self.max_p):
        #        self.max_p = abs(y_p[i])
        #    if(abs(y_a[i]) > self.max_a):
        #        self.max_a = abs(y_a[i])


        self.predicted_p = sum(y_p)/len(y_p)
        self.predicted_a = sum(y_a)/len(y_a)
        self.predicted_d = sum(y_d)/len(y_d)

        if(abs(self.predicted_p) > self.max_p):
            self.max_p = abs(self.predicted_p)
            self.max_p_i = len(self.p_predictions)
        if(abs(self.predicted_a) > self.max_a):
            self.max_a = abs(self.predicted_a)
            self.max_a_i = len(self.a_predictions)
        if(abs(self.predicted_d) > self.max_d):
            self.max_d = abs(self.predicted_d)
            self.max_d_i = len(self.d_predictions)

        if(self.predicted_p > 0):
            self.p_pos_count += 1
        else:
            self.p_neg_count += 1

        if(self.predicted_a > 0):
            self.a_pos_count += 1
        else:
            self.a_neg_count += 1

        if(self.predicted_d > 0):
            self.d_pos_count += 1
        else:
            self.d_neg_count += 1

        self.p_predictions.append(self.predicted_p)
        self.a_predictions.append(self.predicted_a)
        self.d_predictions.append(self.predicted_d)

        return (self.predicted_p, self.predicted_a, self.predicted_d)

    def predict_final_emotion(self):
        largest_p = self.p_predictions[self.max_p_i]
        largest_a = self.a_predictions[self.max_a_i]
        largest_d = self.d_predictions[self.max_d_i]

        if((self.p_pos_count > self.p_neg_count and largest_p > 0) or (self.p_neg_count > self.p_pos_count and largest_p < 0)):
            self.p_predictions.pop(self.max_p_i)
            self.predicted_p = (sum(self.p_predictions)/len(self.p_predictions)) * 0.5 + largest_p * 0.5
        else:
            self.predicted_p = sum(self.p_predictions)/len(self.p_predictions)

        if((self.a_pos_count > self.a_neg_count and largest_a > 0) or (self.a_neg_count > self.a_pos_count and largest_a < 0)):
            self.a_predictions.pop(self.max_a_i)
            self.predicted_a = (sum(self.a_predictions)/len(self.a_predictions)) * 0.5 + largest_a * 0.5
        else:
            self.predicted_a = sum(self.a_predictions)/len(self.a_predictions)

        if((self.d_pos_count > self.d_neg_count and largest_d > 0) or (self.d_neg_count > self.d_pos_count and largest_d < 0)):
            self.d_predictions.pop(self.max_d_i)
            self.predicted_d = (sum(self.d_predictions)/len(self.d_predictions)) * 0.5 + largest_d * 0.5
        else:
            self.predicted_d = sum(self.d_predictions)/len(self.d_predictions)

        return (self.predicted_p, self.predicted_a, self.predicted_d)

    def clear(self):
        self.p_predictions = []
        self.a_predictions = []
        self.d_predictions = []
        
        self.max_p = 0
        self.max_p_i = -1 # index
        self.max_a = 0
        self.max_a_i = -1 # index
        self.max_d = 0
        self.max_d_i = -1 # index

        self.p_pos_count = 0
        self.p_neg_count = 0

        self.a_pos_count = 0
        self.a_neg_count = 0

        self.d_pos_count = 0
        self.d_neg_count = 0

        self.predicted_p = 0.0
        self.predicted_a = 0.0
        self.predicted_d = 0.0