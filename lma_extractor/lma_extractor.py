import numpy as np

class LMAExtractor():
    def __init__(self, engine, outfile = "lma_features", write_to_file=False, pool_rate = 30, label=(0,0,0)):
        self._engine = engine
        self._pooling_rate = pool_rate
        self._frame_counter = 0

        self._outfile = outfile
        open(outfile +'.txt', 'w').close() #Clear file

        self._write_to_file = write_to_file

        self._currentData = []
        
        self._lma_features = []

        self._label = label


    def record_frame(self):
        sim_pose, sim_vel, link_pos, link_orn, vel_dict = self._engine.get_pose_and_links()

        new_frame_data = {}

        pose = [i for i in sim_pose]
        
        timestep = 1 / 30
        pose.insert(0, timestep)

        # Frame Data format: 
        #   {
        #       frame: [frame_data in deepmimic format], 
        #       link_name: [link_position (3D), link_rotation (4D), joint_velocity (1D or 3D)] 
        #   }


        # Link Names:
        #   root: [link_position (3D), link_rotation (4D), [linear_velocity (3D), ang_velocity (3D)]] 
        #   chest: [link_position (3D), link_rotation (4D), linear_velocity (3D)] 
        #   neck: [link_position (3D), link_rotation (4D), linear_velocity (3D)] 
        #   right_hip: [link_position (3D), link_rotation (4D), linear_velocity (3D)] 
        #   right_knee: [link_position (3D), link_rotation (4D), linear_velocity (1D)] 
        #   right_ankle: [link_position (3D), link_rotation (4D), linear_velocity (3D)] 
        #   right_shoulder: [link_position (3D), link_rotation (4D), linear_velocity (3D)] 
        #   right_elbow: [link_position (3D), link_rotation (4D), linear_velocity (1D)] 
        #   right_wrist: [link_position (3D), link_rotation (4D), linear_velocity (0D)] 
        #   left_hip: [link_position (3D), link_rotation (4D), linear_velocity (3D)] 
        #   left_knee: [link_position (3D), link_rotation (4D), linear_velocity (1D)] 
        #   left_ankle: [link_position (3D), link_rotation (4D), linear_velocity (3D)] 
        #   left_shoulder: [link_position (3D), link_rotation (4D), linear_velocity (3D)] 
        #   left_elbow: [link_position (3D), link_rotation (4D), linear_velocity (1D)] 
        #   left_wrist: [link_position (3D), link_rotation (4D), linear_velocity (0D)] 

        new_frame_data["frame"] = pose

        for link in link_pos:
            link_data = [(link_pos[link]), (link_orn[link]), (vel_dict[link])]
            new_frame_data[link] = link_data
            #print(link)
            #print(new_frame_data[link])


        self._currentData.append(new_frame_data) #Add current data
        self._frame_counter += 1

        if(self._frame_counter % self._pooling_rate == 0 and self._frame_counter != 0):
            current_lma_features = self._compute_LMA_features()

            self._lma_features.append(current_lma_features)

            self._currentData = []

            if(self._write_to_file):
                self._append_lma_features()

            return True, current_lma_features

        return False, []

    def get_lma_features(self):
        return self._lma_features

    def reset(self):
        open(self.outfile +'.txt', 'w').close() #Clear file
        self._lma_features = []
        self._currentData = []
        self._frame_counter = 0

    def _compute_LMA_features(self):
        # LMA Data format: 
        #   {
        #       frame_counter: frame at which LMA features were computed, 
        #       label: PAD Emotional Coordinates (3D)
        #       lma_features: [
        #                       average hand_distance (1D), 
        #                       average l_hand_hip_distance (1D), 
        #                       average r_hand_hip_distance (1D),
        #                       average feet_distance (1D),
        #                       average l_hand_chest_distance (1D),
        #                       average r_hand_chest_distance (1D),
        #                       average l_elbow_hip_distance (1D),
        #                       average r_elbow_hip_distance (1D),
        #                       average chest_pelvis_distance (1D),
        #                       average neck_chest_distance (1D),
        #
        #                       average neck_rotation (3D)
        #                       average pelvis_rotation (3D)
        #
        #                       std l_hand_positions (1D)
        #                       std r_hand_positions (1D)
        #
        #                       average l_forearm_velocity (1D)
        #                       average r_forearm_velocity (1D)
        #                       average pelvis_velocity (3D)
        #                       average l_foot_velocity (3D)
        #                       average r_foot_velocity (3D)
        #
        #                       average upper_body_volume (1D)
        #                       average total_body_volume (1D)
        #                       average distane_traveled (1D)
        #                     ] 
        #   }

        # Get LMA features of each frame
        lma_features = []

        hand_distances = []
        l_hand_hip_distances = []
        r_hand_hip_distances = []
        feet_distances = []
        l_hand_chest_distances = []
        r_hand_chest_distances = []
        l_elbow_hip_distances = []
        r_elbow_hip_distances = []
        chest_pelvis_distances = []
        neck_chest_distances = []
        neck_rotations = []
        pelvis_rotations = []
        l_hand_positions = []
        r_hand_positions = []
        l_forearm_velocities = []
        r_forearm_velocities = []
        pelvis_velocities = []
        l_foot_velocities = []
        r_foot_velocities = []
        upper_body_volumes = []
        total_body_volumes = []
        distance_traveled = []



        for data in self._currentData:
            ## Hands Distance
            hand_distances.append(self._compute_distance(data["right_wrist"][0], data["left_wrist"][0]))

            ## Left Hand-Hip Distance
            l_hand_hip_distances.append(self._compute_distance(data["left_wrist"][0], data["root"][0]))

            ## Right Hand-Hip Distance
            r_hand_hip_distances.append(self._compute_distance(data["right_wrist"][0], data["root"][0]))

            ## Feet Distance
            feet_distances.append(self._compute_distance(data["right_ankle"][0], data["left_ankle"][0]))

            ## Left Hand-Chest Distance
            l_hand_chest_distances.append(self._compute_distance(data["left_wrist"][0], data["chest"][0]))

            ## Right Hand-Chest Distance
            r_hand_chest_distances.append(self._compute_distance(data["right_wrist"][0], data["chest"][0]))

            ## Left Elbow-Hip Distance
            l_elbow_hip_distances.append(self._compute_distance(data["left_elbow"][0], data["chest"][0]))

            ## Right Elbow-Hip Distance
            r_elbow_hip_distances.append(self._compute_distance(data["right_elbow"][0], data["chest"][0]))

            ## Chest-Pelvis Distance
            chest_pelvis_distances.append(self._compute_distance(data["chest"][0], data["root"][0]))

            ## Neck-Chest Distance
            neck_chest_distances.append(self._compute_distance(data["neck"][0], data["chest"][0]))
            
            ## Neck Rotation
            neck_rotations.append(data["neck"][1])

            ## Pelvis Rotation
            pelvis_rotations.append(data["root"][1])

            ## Left Hand Position (for std)
            l_hand_positions.append(data["left_wrist"][0])

            ## Right Hand Position (for std)
            r_hand_positions.append(data["right_wrist"][0])

            ## Left Forearm Velocity
            l_forearm_velocities.append(data["left_elbow"][2])

            ## Right Forearm Velocity
            r_forearm_velocities.append(data["right_elbow"][2])

            ## Pelvis Velocity
            pelvis_velocities.append(data["root"][2][0])

            ## Left Foot Velocity
            l_foot_velocities.append(data["left_ankle"][2])

            ## Right Foot Velocity
            r_foot_velocities.append(data["right_ankle"][2])

            ## Upper Body Volume (elbow to elbow, head to pelvis)
            upper_body_volumes.append(self._compute_pyramid_volume(data["left_elbow"][0], data["right_elbow"][0], data["neck"][0], data["root"][0]))

            ## Total Volume (shoulder to shoulder, head to toe)
            total_body_volumes.append(self._compute_box_volume(data["left_shoulder"][0], data["right_shoulder"][0], data["neck"][0], self._compute_midpoint(data["left_ankle"][0], data["right_ankle"][0])))

            ## Average Distance Traveled per frame
            distance_traveled.append(data["root"][0])


        # Compute averages and stuff
        lma_features.append(self._compute_average_distance(hand_distances))
        lma_features.append(self._compute_average_distance(l_hand_hip_distances))
        lma_features.append(self._compute_average_distance(r_hand_hip_distances))
        lma_features.append(self._compute_average_distance(feet_distances))
        lma_features.append(self._compute_average_distance(l_hand_chest_distances))
        lma_features.append(self._compute_average_distance(r_hand_chest_distances))
        lma_features.append(self._compute_average_distance(l_elbow_hip_distances))
        lma_features.append(self._compute_average_distance(r_elbow_hip_distances))
        lma_features.append(self._compute_average_distance(chest_pelvis_distances))
        lma_features.append(self._compute_average_distance(neck_chest_distances))

        lma_features.append(self._compute_average_rotation(neck_rotations))
        lma_features.append(self._compute_average_rotation(pelvis_rotations))

        lma_features.append(self._compute_std_position(l_hand_positions))
        lma_features.append(self._compute_std_position(r_hand_positions))

        lma_features.append(self._compute_average_velocity(l_forearm_velocities))
        lma_features.append(self._compute_average_velocity(r_forearm_velocities))
        lma_features.append(self._compute_average_velocity(pelvis_velocities))
        lma_features.append(self._compute_average_velocity(l_foot_velocities))
        lma_features.append(self._compute_average_velocity(r_foot_velocities))

        lma_features.append(self._compute_average_distance(upper_body_volumes))
        lma_features.append(self._compute_average_distance(total_body_volumes))

        lma_features.append(self._compute_average_distance_traveled(distance_traveled))

        current_lma_features = {}
        current_lma_features["frame_counter"] = self._frame_counter
        current_lma_features["label"] = self._label
        current_lma_features["lma_features"] = lma_features #todo: replace this to compute actual lma_features
        return current_lma_features

    ### MATH METHODS ###

    def _compute_midpoint(self, pos_1, pos_2):
        mid = tuple([sum(x)/2.0 for x in zip(pos_1,pos_2)])
        return mid

    def _compute_distance(self, pos_1, pos_2):
        p1 = np.array(pos_1)
        p2 = np.array(pos_2)

        squared_dist = np.sum((p1-p2)**2, axis=0)
        
        return np.sqrt(squared_dist)

    def _compute_pyramid_volume(self, left_max, right_max, base, tip):
        width = self._compute_distance(left_max, right_max)
        height = self._compute_distance(base, tip)

        return (width * width * height)/3.0

    def _compute_box_volume(self, left_max, right_max, base, tip):
        width = self._compute_distance(left_max, right_max)
        height = self._compute_distance(base, tip)
        
        return width * width * height

    def _compute_average_distance(self, distances):
        return sum(distances)/len(distances)

    def _compute_average_rotation(self, rotations):
        avg_rotation = tuple([sum(x)/len(rotations) for x in zip(*rotations)])

        return avg_rotation

    def _compute_average_distance_traveled(self, positions):
        distances_traveled = []
        for i in range(1, len(positions)):
            distances_traveled.append(self._compute_distance(positions[i], positions[i-1]))

        return sum(distances_traveled)/len(distances_traveled)

    def _compute_average_velocity(self, velocities):
        avg_pos = tuple([sum(x)/len(velocities) for x in zip(*velocities)])

        return avg_pos

    def _compute_std_position(self, positions):
        arr = np.array(positions)
        return np.std(arr)

    ### FILE WRITING METHODS ###

    def _append_lma_features(self):
        # Appends the last entry of the lma_features array to a file
        with open(self._outfile + ".txt", 'a') as wh:
            wh.write(str(self._lma_features[-1]) + "\n")

        return

    def write_lma_features(self):
        # Writes entire lma feature array to a file
        with open(self._outfile + ".txt", 'w') as wh:
            for frame in self._lma_features:
                wh.write(str(frame) + "\n")

        return

    def write_lma_features_mult(self):
        # Writes each entry of the lma feature array to a different file
        for frame in self._lma_features:
            with open(self._outfile + "_" + str(frame['frame_number']) + ".txt", 'w') as wh:
                wh.write(str(frame) + "\n")
    
        return