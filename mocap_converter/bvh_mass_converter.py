from bvhtomimic import BvhConverter
import os

import shutil

# This script converts all BVH files in the dance_emotions, kin_emotions and walk_emotions directories into the Deepmimic
# friendly format

walk_directory = 'bvh_data/walk_emotions'
dance_directory = 'bvh_data/dance_emotions'
dance_directory2 = 'bvh_data/dance_emotions_extra'
kin_directory = 'bvh_data/kin_emotions'
kin_directory2 = 'bvh_data/kin_emotions_full'

walk_settings = 'settings/walk_settings.json'
dance_settings = 'settings/dance_settings.json'
kin_settings = 'settings/kin_settings.json'

walk_output = 'deepmimic_data/walk'
dance_output = 'deepmimic_data/dance'
dance_output2 = 'deepmimic_data/dance_extra'
kin_output = 'deepmimic_data/kin'
kin_output2 = 'deepmimic_data/kin_full'

kin_conv_dict = {"A": "angry", "D": "disgusted", "F": "afraid", "H": "happy", "N": "neutral", "S": "sad"}
kin_count = {"A": 0, "D": 0, "F": 0, "H": 0, "N": 0, "S": 0}
 

meta_file_path = "deepmimic_data/META_KIN_FULL.txt"
files = []


# Read  META File
with open(meta_file_path, 'r') as r:
    for file in r.readlines():
        files.append(file.replace("\n",""))

# Overwrite META File
meta_file = open(meta_file_path, "a")

# Walk
"""
for filename in os.listdir(walk_directory):
    f = os.path.join(walk_directory, filename)
    if os.path.isfile(f):
        print(f)

        # File args: emotion, index, type
        #file_args = filename.split("_")
        #file_args[2] = file_args[2].split(".")[0]

        converter = BvhConverter(walk_settings)

        o = os.path.join(walk_output, filename.replace(".bvh", ".txt"))

        if(os.path.exists(o) or o in files):
            continue

        meta_file.write(o + "\n")
        converter.writeDeepMimicFile(f, o)


# Dance
for filename in os.listdir(dance_directory):
    f = os.path.join(dance_directory, filename)
    if os.path.isfile(f):
        print(f)

        # File args: emotion, index, type
        #file_args = filename.split("_")
        #file_args[2] = file_args[2].split(".")[0]

        converter = BvhConverter(dance_settings)

        o = os.path.join(dance_output, filename.replace(".bvh", ".txt"))

        if(os.path.exists(o) or o in files):
            continue
        
        meta_file.write(o + "\n")
        converter.writeDeepMimicFile(f, o)
"""

# Kin
for child_directory in os.listdir(kin_directory2):
    path_to_child_directory = os.path.join(kin_directory2, child_directory)
    if(child_directory == "META"):
        continue

    for filename in os.listdir(path_to_child_directory):
        f = os.path.join(path_to_child_directory, filename)
        if os.path.isfile(f):
            print(f)

            emotion = filename[3]

            output_filename = kin_conv_dict[emotion] + "_kin_" + str(kin_count[emotion]) + ".txt"
            
            kin_count[emotion] = kin_count[emotion] + 1

            converter = BvhConverter(kin_settings)

            o = os.path.join(kin_output2, output_filename)

            if(os.path.exists(o) or o in files):
                continue
            
            meta_file.write(o + "\n")
            converter.writeDeepMimicFile(f, o)

            #shutil.copyfile(o, os.path.join("../lma_extractor/lma_features/new", output_filename)) ## Feed directly into lma_extractor

# Dance 2
"""
for filename in os.listdir(dance_directory2):
    f = os.path.join(dance_directory2, filename)
    if os.path.isfile(f):
        print(f)

        # File args: emotion, index, type
        #file_args = filename.split("_")
        #file_args[2] = file_args[2].split(".")[0]

        converter = BvhConverter(dance_settings)

        o = os.path.join(dance_output2, filename.replace(".bvh", ".txt"))

        if(os.path.exists(o) or o in files):
            continue
        
        meta_file.write(o + "\n")
        converter.writeDeepMimicFile(f, o)

        shutil.copyfile(o, os.path.join("../lma_extractor/mocap_data", filename.replace(".bvh", ".txt"))) ## Feed directly into lma_extractor
"""

meta_file.close()