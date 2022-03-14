import numpy as np
import os.path

import pandas as pd

import xgboost as xgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


xgb.set_config(verbosity=0)

class EmotionClassifier():
    def __init__(self):
        self._model_p = xgb.XGBRegressor(verbosity=0)
        self._model_p.load_model("../emotion_classifier/l2p_model_final.json")

        self._model_a = xgb.XGBRegressor(verbosity=0)
        self._model_a.load_model("../emotion_classifier/l2a_model_final.json")

        self.predicted_p = 0.0
        self.predicted_a = 0.0

    def predict_emotion_coordinates(self, lma_features):    
        p_predictions = []
        a_predictions = []
        
        normalizer = MinMaxScaler(feature_range=(0, 1), copy=True)

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
        normalizer.fit(rows)
        rows = normalizer.transform(rows)

        df = pd.DataFrame(rows, columns=['avg_hand_distance', 'avg_l_hand_hip_distance', 'avg_r_hand_hip_distance', 'avg_feet_distance', 'avg_l_hand_chest_distance', 'avg_r_hand_chest_distance', 'avg_l_elbow_hip_distance', 'avg_r_elbow_hip_distance', 'avg_chest_pelvis_distance', 'avg_neck_chest_distance', 'avg_neck_rotation_w', 'avg_neck_rotation_x', 'avg_neck_rotation_y', 'avg_neck_rotation_z',
         'avg_pelvis_rotation_w', 'avg_pelvis_rotation_x', 'avg_pelvis_rotation_y', 'avg_pelvis_rotation_z', 'std_l_hand_position', 'std_r_hand_position', 'avg_l_forearm_velocity', 
         'avg_r_forearm_velocity', 'avg_pelvis_velocity_x', 'avg_pelvis_velocity_y', 'avg_pelvis_velocity_z', 'avg_l_foot_velocity_x', 
         'avg_l_foot_velocity_y', 'avg_l_foot_velocity_z', 'avg_r_foot_velocity_x', 'avg_r_foot_velocity_y', 'avg_r_foot_velocity_z', 'avg_upper_body_volume', 
         'avg_distance_traveled'])


        pos_p = 0
        neg_p = 0
        pos_a = 0
        neg_a = 0

        for i in range(len(rows)):
            row=df.iloc[i]
            #print(row)
            
            x = np.asarray([row])

            y_p = self._model_p.predict(x)
            y_a =  self._model_a.predict(x)
            
            """
            if y_p[0] > 0:
                pos_p +=1
            else:
                neg_p += 1

            if y_a[0] > 0:
                pos_a +=1
            else:
                neg_a += 1
            """

            p_predictions.append(y_p[0])
            a_predictions.append(y_a[0])

        """
        for i in range(len(p_predictions)):
            if(pos_p > neg_p):
                if(p_predictions[i] < 0):
                    p_predictions[i] = 0.0
            if(pos_p < neg_p):
                if(p_predictions[i] > 0):
                    p_predictions[i] = 0.0

            if(pos_a > neg_a):
                if(a_predictions[i] < 0):
                    a_predictions[i] = 0.0
            if(pos_a < neg_a):
                if(a_predictions[i] > 0):
                    a_predictions[i] = 0.0
        """

        self.predicted_p = sum(p_predictions)/len(p_predictions)
        self.predicted_a = sum(a_predictions)/len(a_predictions)

        return (self.predicted_p, self.predicted_a)