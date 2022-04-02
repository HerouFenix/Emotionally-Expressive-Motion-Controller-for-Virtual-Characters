import numpy as np
import os.path

class LMAExtractor():
    def __init__(self, engine, outfile = "lma_features", append_to_file=False, pool_rate = 30, label=(0,0,0), ignore_amount = 0, round_values=False):
        self._engine = engine
        self._pooling_rate = pool_rate
        self._frame_counter = 0

        self._outfile = outfile
        if(append_to_file and os.path.exists(outfile)):
            open(outfile, 'w').close() #Clear file

        self._append_to_file = append_to_file

        self._currentData = []
        
        self._lma_features = []

        self._label = label

        self._ignore_amount = ignore_amount
        self._number_ignored = 0

        self._round_values = round_values

    def record_frame(self):
        sim_pose, sim_vel, link_pos, link_orn, vel_dict = self._engine.get_pose_and_links()

        new_frame_data = {}

        pose = [i for i in sim_pose]
        
        timestep = 1 / 30 # Assuming all our animations run at 30 fps which is wrong. But since we don't ever use this it doesnt really matter
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

            if(self._number_ignored < self._ignore_amount): # If we want to ignore the first X set of frames
                self._number_ignored += self._pooling_rate
                self._currentData = []
                return False, []

            current_lma_features = self._compute_LMA_features()
            self._currentData = []

            self._lma_features.append(current_lma_features)
            
            if(self._append_to_file):
                self._append_lma_features()


            return True, current_lma_features

        return False, []

    def get_lma_features(self):
        return self._lma_features

    def reset(self):
        if(self._append_to_file and os.path.exists(self._outfile)):
            open(self._outfile, 'w').close() #Clear file
        self._lma_features = []
        self._currentData = []
        self._frame_counter = 0

    def clear(self):
        self._lma_features = []
        self._currentData = []

    def _compute_LMA_features(self):
        # LMA Data format: 
        #   {
        #       frame_counter: frame at which LMA features were computed, 
        #       label: PAD Emotional Coordinates (3D)
        #       lma_features: [
        # Position Features
        #                       max hand_distance (1D), 
        #                       average l_hand_hip_distance (1D), 
        #                       average r_hand_hip_distance (1D),
        #                       max stride length (distance between left and right foot) (1D),
        #                       average l_hand_chest_distance (1D),
        #                       average r_hand_chest_distance (1D),
        #                       average l_elbow_hip_distance (1D),
        #                       average r_elbow_hip_distance (1D),
        #                       average chest_pelvis_distance (1D),
        #                       average neck_chest_distance (1D),
        #
        #                       average neck_rotation (3D)
        #
        #                       average total_body_volume (1D)
        #
        #                       triangle area between hands and neck (1D)
        #                       triangle area between feet and root (1D)
        #
        #
        # Movement Features:
        #
        #                       average l_hand speed (1D)
        #                       average r_hand speed (1D)
        #                       average l_foot_speed (1D)
        #                       average r_foot_speed (1D)                 
        #                       average neck speed (1D)
        #                       
        #                       average l_hand acceleration magnitude (1D)
        #                       average r_hand acceleration magnitude (1D)
        #                       average l_foot acceleration magnitude (1D)
        #                       average r_foot acceleration magnitude (1D)                 
        #                       average neck acceleration magnitude (1D)
        # 
        #                       average l_hand movement jerk (3D)
        #                       average r_hand movement jerk (3D)
        #                       average l_foot movement jerk (3D)
        #                       average r_foot movement jerk (3D)                  
        #                       average head movement jerk (3D)
        #                     ] 
        #   }

        # Get LMA features of each frame
        lma_features = []

        hand_distances = []
        l_hand_hip_distances = []
        r_hand_hip_distances = []
        stride_distances = []
        l_hand_chest_distances = []
        r_hand_chest_distances = []
        l_elbow_hip_distances = []
        r_elbow_hip_distances = []
        chest_pelvis_distances = []
        neck_chest_distances = []

        neck_rotations = []

        total_body_volumes = []

        hands_neck_triangles = []
        feet_root_triangles = []

        l_hand_positions = []
        r_hand_positions = []
        l_foot_positions = []
        r_foot_positions = []
        neck_positions = []

        l_foot_velocities = []
        r_foot_velocities = []
        neck_velocities = []


        for data in self._currentData:
            # == POSITIONAL FEATURES ==

            ## Hands Distance
            hand_distances.append(self._compute_distance(data["right_wrist"][0], data["left_wrist"][0]))

            ## Left Hand-Hip Distance
            l_hand_hip_distances.append(self._compute_distance(data["left_wrist"][0], data["root"][0]))

            ## Right Hand-Hip Distance
            r_hand_hip_distances.append(self._compute_distance(data["right_wrist"][0], data["root"][0]))

            ## Stride Distances
            stride_distances.append(self._compute_distance(data["right_ankle"][0], data["left_ankle"][0]))

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


            ## Total Body Volumes
            positions = [data[link][0] for link in data.keys() if link != "frame"]
            total_body_volumes.append(self._compute_box_volume(positions))


            ## Hands-Neck Triangle Area
            hands_neck_triangles.append(self._compute_triangle_area(data["left_wrist"][0], data["right_wrist"][0], data["neck"][0]))


            ## Feet-Root Triangle Area
            feet_root_triangles.append(self._compute_triangle_area(data["left_ankle"][0], data["right_ankle"][0], data["root"][0]))



            # == MOVEMENT FEATURES ==
            ## Left Hand Position (for speed and velocity)
            l_hand_positions.append(data["left_wrist"][0])

            ## Right Hand Position (for speed and velocity)
            r_hand_positions.append(data["right_wrist"][0])

            ## Left Foot Position (for speed)
            l_foot_positions.append(data["left_ankle"][0])

            ## Right Foot Position (for speed)
            r_foot_positions.append(data["right_ankle"][0])

            ## Neck Position (for speed)
            neck_positions.append(data["neck"][0])
            
            ## Left Foot Velocity
            l_foot_velocities.append(data["left_ankle"][2])

            ## Right Foot Velocity
            r_foot_velocities.append(data["right_ankle"][2])

            ## Neck Velocity
            neck_velocities.append(data["neck"][2])


        # Compute averages and stuff
        l_hand_velocities = self._compute_velocities_from_positions(l_hand_positions, data["frame"][0]) # TODO: Check that this is correct. Somewhy some of the computed values are SUPER high
        r_hand_velocities = self._compute_velocities_from_positions(r_hand_positions, data["frame"][0]) # TODO: Check that this is correct. Somewhy some of the computed values are SUPER high

        l_hand_accelerations = self._compute_accelerations_from_velocities(l_hand_velocities, data["frame"][0])
        r_hand_accelerations = self._compute_accelerations_from_velocities(r_hand_velocities, data["frame"][0])
        l_foot_accelerations = self._compute_accelerations_from_velocities(l_foot_velocities, data["frame"][0])
        r_foot_accelerations = self._compute_accelerations_from_velocities(r_foot_velocities, data["frame"][0])
        neck_accelerations = self._compute_accelerations_from_velocities(neck_velocities, data["frame"][0])

        l_hand_accelerations_magn = [np.linalg.norm(np.array(accel)) for accel in l_hand_accelerations]
        r_hand_accelerations_magn = [np.linalg.norm(np.array(accel)) for accel in r_hand_accelerations]
        l_foot_accelerations_magn = [np.linalg.norm(np.array(accel)) for accel in l_foot_accelerations]
        r_foot_accelerations_magn = [np.linalg.norm(np.array(accel)) for accel in r_foot_accelerations]
        neck_accelerations_magn = [np.linalg.norm(np.array(accel)) for accel in neck_accelerations]

        l_hand_speed = self._compute_speed_from_positions(l_hand_positions, data["frame"][0])
        r_hand_speed = self._compute_speed_from_positions(r_hand_positions, data["frame"][0])
        l_foot_speed = self._compute_speed_from_positions(l_foot_positions, data["frame"][0])
        r_foot_speed = self._compute_speed_from_positions(r_foot_positions, data["frame"][0])
        neck_speed = self._compute_speed_from_positions(neck_positions, data["frame"][0])

        l_hand_jerks = self._compute_jerk_from_accelerations(l_hand_accelerations, data["frame"][0])
        r_hand_jerks = self._compute_jerk_from_accelerations(r_hand_accelerations, data["frame"][0])
        l_foot_jerks = self._compute_jerk_from_accelerations(l_foot_accelerations, data["frame"][0])
        r_foot_jerks = self._compute_jerk_from_accelerations(r_foot_accelerations, data["frame"][0])
        neck_jerks = self._compute_jerk_from_accelerations(neck_accelerations, data["frame"][0])
        

        ## == POSITION FEATURES ==
        lma_features.append(self._compute_max_distance(hand_distances))
        lma_features.append(self._compute_average_distance(l_hand_hip_distances))
        lma_features.append(self._compute_average_distance(r_hand_hip_distances))
        lma_features.append(self._compute_max_distance(stride_distances))
        lma_features.append(self._compute_average_distance(l_hand_chest_distances))
        lma_features.append(self._compute_average_distance(r_hand_chest_distances))
        lma_features.append(self._compute_average_distance(l_elbow_hip_distances))
        lma_features.append(self._compute_average_distance(r_elbow_hip_distances))
        lma_features.append(self._compute_average_distance(chest_pelvis_distances))
        lma_features.append(self._compute_average_distance(neck_chest_distances))

        lma_features.append(self._compute_average_rotation(neck_rotations))

        lma_features.append(self._compute_average_distance(total_body_volumes))

        lma_features.append(self._compute_average_distance(hands_neck_triangles))
        lma_features.append(self._compute_average_distance(feet_root_triangles))


        ## == MOVEMENT FEATURES ==
        # Speed
        lma_features.append(self._compute_average_distance(l_hand_speed))
        lma_features.append(self._compute_average_distance(r_hand_speed))
        lma_features.append(self._compute_average_distance(l_foot_speed))
        lma_features.append(self._compute_average_distance(r_foot_speed))
        lma_features.append(self._compute_average_distance(neck_speed))


        # Acceleration Magnitude
        lma_features.append(self._compute_average_distance(l_hand_accelerations_magn))
        lma_features.append(self._compute_average_distance(r_hand_accelerations_magn))
        lma_features.append(self._compute_average_distance(l_foot_accelerations_magn))
        lma_features.append(self._compute_average_distance(r_foot_accelerations_magn))
        lma_features.append(self._compute_average_distance(neck_accelerations_magn))

        # Movement Jerk
        lma_features.append(self._compute_average_rotation(l_hand_jerks))
        lma_features.append(self._compute_average_rotation(r_hand_jerks))
        lma_features.append(self._compute_average_rotation(l_foot_jerks))
        lma_features.append(self._compute_average_rotation(r_foot_jerks))
        lma_features.append(self._compute_average_rotation(neck_jerks))


        if(self._round_values):
            for i in range(0,len(lma_features)):
                if(type(lma_features[i]) is tuple):
                    new_tuple = ()
                    for j in range(0, len(lma_features[i])):
                        new_tuple_add = (round(lma_features[i][j],10),)
                        new_tuple = new_tuple + new_tuple_add

                    lma_features[i] = new_tuple 
                else:
                    lma_features[i] = round(lma_features[i],10)

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

    def _compute_triangle_area(self, point_1, point_2, point_3):
        a = self._compute_distance(point_1, point_2)
        b = self._compute_distance(point_2, point_3)
        c = self._compute_distance(point_3, point_1)

        s = (a+b+c) * 0.5
        area = (s*(s-a) * (s-b)*(s-c)) ** 0.5 
        return area

    def _compute_pyramid_volume(self, left_max, right_max, base, tip):
        width = self._compute_distance(left_max, right_max)
        height = self._compute_distance(base, tip)

        return (width * width * height)/3.0

    def _compute_box_volume(self, positions):
        # Get left-most, bottom-most, back-most position
        min = [float('inf'), float('inf'), float('inf')]

        # Get right-most, top-most, front-most position
        max = [float('-inf'), float('-inf'), float('-inf')]

        for position in positions:
            if position[0] < min[0]: min[0] = position[0]
            if position[1] < min[1]: min[1] = position[1]
            if position[2] < min[2]: min[2] = position[2]

            if position[0] > max[0]: max[0] = position[0]
            if position[1] > max[1]: max[1] = position[1]
            if position[2] > max[2]: max[2] = position[2]
        
        width = abs(max[0] - min[0])
        height = abs(max[1] - min[1])
        depth = abs(max[2] - min[2])

        return width * height * depth

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

    def _compute_max_distance(self, distances):
        return max(distances)

    def _compute_velocities_from_positions(self, positions, frame_duration):
        velocities = []
        for i in range(1, len(positions)):
            velocity_x = (positions[i][0] - positions[i-1][0]) / frame_duration
            velocity_y = (positions[i][1] - positions[i-1][1]) / frame_duration
            velocity_z = (positions[i][2] - positions[i-1][2]) / frame_duration

            velocities.append((velocity_x, velocity_y, velocity_z))

        return velocities


    def _compute_accelerations_from_velocities(self, velocities, frame_duration):
        accelerations = []
        for i in range(1, len(velocities)):
            acceleration_x = (velocities[i][0] - velocities[i-1][0]) / frame_duration
            acceleration_y = (velocities[i][1] - velocities[i-1][1]) / frame_duration
            acceleration_z = (velocities[i][2] - velocities[i-1][2]) / frame_duration

            accelerations.append((acceleration_x, acceleration_y, acceleration_z))

        return velocities


    def _compute_speed_from_positions(self, positions, frame_duration):
        speeds = []
        for i in range(1, len(positions)):
            distance = self._compute_distance(positions[i], positions[i-1])
            speed = distance / frame_duration

            speeds.append(speed)

        return speeds

    def _compute_jerk_from_accelerations(self, accelerations, frame_duration):
        jerks = []
        for i in range(1, len(accelerations)):
            jerk_x = (accelerations[i][0] - accelerations[i-1][0]) / frame_duration
            jerk_y = (accelerations[i][1] - accelerations[i-1][1]) / frame_duration
            jerk_z = (accelerations[i][2] - accelerations[i-1][2]) / frame_duration

            jerks.append((jerk_x, jerk_y, jerk_z))

        return jerks


    ### FILE WRITING METHODS ###

    def _append_lma_features(self):
        # Appends the last entry of the lma_features array to a file
        with open(self._outfile, 'a') as wh:
            wh.write(str(self._lma_features[-1]) + "\n")

        return

    def write_lma_features(self):
        # Writes entire lma feature array to a file
        with open(self._outfile, 'w') as wh:
            for frame in self._lma_features:
                wh.write(str(frame) + "\n")

        return

    def write_lma_features_mult(self):
        # Writes each entry of the lma feature array to a different file
        for frame in self._lma_features:
            with open(self._outfile + "_" + str(frame['frame_number']), 'w') as wh:
                wh.write(str(frame) + "\n")
    
        return