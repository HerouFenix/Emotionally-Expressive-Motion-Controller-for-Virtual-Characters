import numpy as np
import math
import os.path

class LMAExtractor():
    def __init__(self, engine, frame_duration, outfile = "lma_features", append_to_file=False, pool_rate = 0.5, label=(0,0,0), ignore_amount = 0, round_values=False, write_mocap=False, write_mocap_file = ''):
        self._engine = engine

        self._frame_duration = frame_duration
        self._fps = 1/self._frame_duration

        self._time_between_recordings = pool_rate

        if(pool_rate == -1):
            self._pooling_rate = 5
        else:
            self._pooling_rate = math.floor(self._fps * pool_rate) 

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

        self._last_velocities = [(0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0)] # Used when computing accelerations (l_hand, r_hand, l_foot, r_foot, neck)
        self._last_accelerations = [(0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0)] # Used when computing accelerations (l_hand, r_hand, l_foot, r_foot, neck)

        self._write_mocap = write_mocap
        self._write_mocap_file = write_mocap_file
        if(write_mocap and os.path.exists(write_mocap_file)):
            open(write_mocap_file, 'w').close() #Clear file

        self.mocap_full = [] 
        self.lma_full = []

    def record_frame(self):
        sim_pose, sim_vel, link_pos, link_orn, vel_dict = self._engine.get_pose_and_links()

        pose = [i for i in sim_pose]

        new_frame_data = {}

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

        if(self._write_mocap):
            self._append_mocap(new_frame_data)
        else:
            self.mocap_full.append(new_frame_data)

        self._currentData.append(new_frame_data) #Add current data
        self._frame_counter += 1

        if(self._frame_counter % self._pooling_rate == 0 and self._frame_counter != 0):

            current_lma_features = self._compute_LMA_features()

            if(self._number_ignored < self._ignore_amount): # If we want to ignore the first X set of frames, then compute the current lma features (to get last velocities and accelerations and stuff), but dont append to lma feature array
                self._number_ignored += self._pooling_rate
                self._currentData = []
                return False, []

            self._currentData = []

            self._lma_features.append(current_lma_features)
            self.lma_full.append(current_lma_features)
            
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
        self._first = True

    def clear_full(self):
        self.lma_full = []
        self.mocap_full = []

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
        #                       average total_body_volume (1D)
        #
        #                       average lower_body_volume (1D) 
        #                       average upper_body_volume (1D) 
        #
        #                       triangle area between hands and neck (1D)
        #                       triangle area between feet and root (1D)
        #
        #
        # Movement Features:
        #
        #                       l_hand speed (1D)
        #                       r_hand speed (1D)
        #                       l_foot_speed (1D)
        #                       r_foot_speed (1D)                 
        #                       neck speed (1D)
        #                       
        #                       l_hand acceleration magnitude (1D)
        #                       r_hand acceleration magnitude (1D)
        #                       l_foot acceleration magnitude (1D)
        #                       r_foot acceleration magnitude (1D)                 
        #                       neck acceleration magnitude (1D)
        # 
        #
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
        upper_body_volumes = []
        lower_body_volumes = []

        hands_neck_triangles = []
        feet_root_triangles = []

        l_hand_positions = []
        r_hand_positions = []
        l_foot_positions = []
        r_foot_positions = []
        neck_positions = []

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

            ## Upper Body Volumes
            positions = [data[link][0] for link in ["root", "chest", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow", "left_wrist",]]
            upper_body_volumes.append(self._compute_box_volume(positions))

            ## Lower Body Volumes
            positions = [data[link][0] for link in ["root", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle"]]
            lower_body_volumes.append(self._compute_box_volume(positions))


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


        # Compute averages and stuff
        l_hand_velocity = self._compute_velocities_from_positions(l_hand_positions) 
        r_hand_velocity = self._compute_velocities_from_positions(r_hand_positions) 
        l_foot_velocity = self._compute_velocities_from_positions(l_foot_positions) 
        r_foot_velocity = self._compute_velocities_from_positions(r_foot_positions) 
        neck_velocity = self._compute_velocities_from_positions(neck_positions) 

        l_hand_acceleration = self._compute_accelerations_from_velocities(l_hand_velocity, self._last_velocities[0])
        r_hand_acceleration = self._compute_accelerations_from_velocities(r_hand_velocity, self._last_velocities[1])
        l_foot_acceleration = self._compute_accelerations_from_velocities(l_foot_velocity, self._last_velocities[2])
        r_foot_acceleration = self._compute_accelerations_from_velocities(r_foot_velocity, self._last_velocities[3])
        neck_acceleration = self._compute_accelerations_from_velocities(neck_velocity, self._last_velocities[4])

        l_hand_acceleration_magn = np.linalg.norm(np.array(l_hand_acceleration))
        r_hand_acceleration_magn = np.linalg.norm(np.array(r_hand_acceleration))
        l_foot_acceleration_magn = np.linalg.norm(np.array(l_foot_acceleration))
        r_foot_acceleration_magn = np.linalg.norm(np.array(r_foot_acceleration))
        neck_acceleration_magn = np.linalg.norm(np.array(neck_acceleration))

        l_hand_speed = self._compute_speed_from_positions(l_hand_positions)
        r_hand_speed = self._compute_speed_from_positions(r_hand_positions)
        l_foot_speed = self._compute_speed_from_positions(l_foot_positions)
        r_foot_speed = self._compute_speed_from_positions(r_foot_positions)
        neck_speed = self._compute_speed_from_positions(neck_positions)

        #l_hand_jerk = self._compute_jerk_from_accelerations(l_hand_acceleration, self._last_accelerations[0])
        #r_hand_jerk = self._compute_jerk_from_accelerations(r_hand_acceleration, self._last_accelerations[1])
        #l_foot_jerk = self._compute_jerk_from_accelerations(l_foot_acceleration, self._last_accelerations[2])
        #r_foot_jerk = self._compute_jerk_from_accelerations(r_foot_acceleration, self._last_accelerations[3])
        #neck_jerk = self._compute_jerk_from_accelerations(neck_acceleration, self._last_accelerations[4])

        # Update Last Velocities
        self._last_velocities = [l_hand_velocity, r_hand_velocity, l_foot_velocity, r_foot_velocity, neck_velocity]
        self._last_accelerations = [l_hand_acceleration, r_hand_acceleration, l_foot_acceleration, r_foot_acceleration, neck_acceleration]

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
        lma_features.append(self._compute_average_distance(upper_body_volumes))
        lma_features.append(self._compute_average_distance(lower_body_volumes))

        lma_features.append(self._compute_average_distance(hands_neck_triangles))
        lma_features.append(self._compute_average_distance(feet_root_triangles))


        ## == MOVEMENT FEATURES ==
        # Speed
        lma_features.append(l_hand_speed)
        lma_features.append(r_hand_speed)
        lma_features.append(l_foot_speed)
        lma_features.append(r_foot_speed)
        lma_features.append(neck_speed)


        # Acceleration Magnitude
        lma_features.append(l_hand_acceleration_magn)
        lma_features.append(r_hand_acceleration_magn)
        lma_features.append(l_foot_acceleration_magn)
        lma_features.append(r_foot_acceleration_magn)
        lma_features.append(neck_acceleration_magn)

        # Movement Jerk
        #lma_features.append(l_hand_jerk)
        #lma_features.append(r_hand_jerk)
        #lma_features.append(l_foot_jerk)
        #lma_features.append(r_foot_jerk)
        #lma_features.append(neck_jerk)



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

    def _compute_velocities_from_positions(self, positions):        
        velocity_x = (positions[len(positions)-1][0] - positions[0][0]) / self._time_between_recordings
        velocity_y = (positions[len(positions)-1][1] - positions[0][1]) / self._time_between_recordings
        velocity_z = (positions[len(positions)-1][2] - positions[0][2]) / self._time_between_recordings

        return (velocity_x, velocity_y, velocity_z)


    def _compute_accelerations_from_velocities(self, velocities_end, velocities_start):
        acceleration_x = (velocities_end[0] - velocities_start[0]) / self._time_between_recordings
        acceleration_y = (velocities_end[1] - velocities_start[1]) / self._time_between_recordings
        acceleration_z = (velocities_end[2] - velocities_start[2]) / self._time_between_recordings

        return (acceleration_x, acceleration_y, acceleration_z)


    def _compute_speed_from_positions(self, positions):
        distance = self._compute_distance(positions[len(positions)-1], positions[0])
        speed = distance / self._time_between_recordings

        return speed

    def _compute_jerk_from_accelerations(self, acceleration_end, acceleration_start):
        jerk_x = (acceleration_end[0] - acceleration_start[0]) / self._time_between_recordings
        jerk_y = (acceleration_end[1] - acceleration_start[1]) / self._time_between_recordings
        jerk_z = (acceleration_end[2] - acceleration_start[2]) / self._time_between_recordings

        return (jerk_x, jerk_y, jerk_z)


    ### FILE WRITING METHODS ###

    def _append_lma_features(self):
        # Appends the last entry of the lma_features array to a file
        with open(self._outfile, 'a') as wh:
            wh.write(str(self._lma_features[-1]) + "\n")

        return

    def _append_mocap(self, mocap_data):
        # Appends the last entry of the lma_features array to a file
        with open(self._write_mocap_file, 'a') as wh:
            wh.write(str(mocap_data) + "\n")

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