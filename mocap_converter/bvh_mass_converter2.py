from bvhtomimic import BvhConverter
import os

import shutil

# This script converts all BVH files in the bandai directories into the Deepmimic
# friendly format

bandai_directory_1 = 'bvh_data/bandai/dataset/1/data/'
bandai_directory_2 = 'bvh_data/bandai/dataset/2/data/'

bandai_settings = 'settings/bandai_settings.json'

bandai_output_1 = 'deepmimic_data/bandai_1'
bandai_output_2 = 'deepmimic_data/bandai_2'

bandai_emotions_1 = {"normal": "neutral", "tired": "tired", "angry": "angry", "happy": "happy", "sad": "sad", "proud": "proud", "not-confident": "afraid", "old": "exhausted", "giant": "confident", "musical": "elated", "active":"active", "masculine":"confident_2", "masculinity":"confident_3"}
bandai_count_1 = {"normal": 0, "tired": 0, "angry": 0, "happy": 0, "sad": 0, "proud": 0, "not-confident": 0, "old": 0, "giant": 0, "musical": 0, "active": 0, "masculine": 0, "masculinity": 0}

bandai_emotions_2 = {"normal": "neutral", "youthful": "happy","exhausted":"exhausted_2", "elderly": "exhausted", "active":"active", "masculine":"confident"}
bandai_count_2 = {"normal": 0, "youthful": 0, "elderly": 0, "active": 0, "masculine": 0, "exhausted": 0}

meta_file_path_1 = "deepmimic_data/META_BANDAI_1.txt"
files = []


# Read  META File
with open(meta_file_path_1, 'r') as r:
    for file in r.readlines():
        files.append(file.replace("\n",""))

# Overwrite META File
meta_file_1 = open(meta_file_path_1, "a")

# Bandai 1
for filename in os.listdir(bandai_directory_1):
    f = os.path.join(bandai_directory_1, filename)
    if os.path.isfile(f):
        print(f)
        filename = filename.split("_")
        motion = filename[1]
        emotion = filename[2]

        if(emotion == "feminine" or emotion == "chimpira" or emotion == "childish"):
            continue
        
        output_filename = bandai_emotions_1[emotion] + "_bandai1_" + str(bandai_count_1[emotion]) + "_" + motion + ".txt"
        #print(output_filename)
        
        bandai_count_1[emotion] = bandai_count_1[emotion] + 1

        converter = BvhConverter(bandai_settings)
        o = os.path.join(bandai_output_1, output_filename)
        if(os.path.exists(o) or o in files):
            continue
        
        meta_file_1.write(o + "\n")
        converter.writeDeepMimicFile(f, o)

meta_file_1.close()


meta_file_path_1 = "deepmimic_data/META_BANDAI_1.txt"
files = []


# Read  META File
meta_file_path_2 = "deepmimic_data/META_BANDAI_2.txt"

with open(meta_file_path_2, 'r') as r:
    for file in r.readlines():
        files.append(file.replace("\n",""))

# Overwrite META File
meta_file_2 = open(meta_file_path_2, "a")

# Bandai 2
for filename in os.listdir(bandai_directory_2):
    f = os.path.join(bandai_directory_2, filename)
    if os.path.isfile(f):
        print(f)
        filename = filename.split("_")
        motion = filename[1]
        emotion = filename[2]

        if(emotion == "feminine" or emotion == "chimpira" or emotion == "childish"):
            continue
        
        output_filename = bandai_emotions_2[emotion] + "_bandai2_" + str(bandai_count_2[emotion]) + "_" + motion + ".txt"
        #print(output_filename)
        
        bandai_count_2[emotion] = bandai_count_2[emotion] + 1

        converter = BvhConverter(bandai_settings)
        o = os.path.join(bandai_output_2, output_filename)
        if(os.path.exists(o) or o in files):
            continue
        
        meta_file_2.write(o + "\n")
        converter.writeDeepMimicFile(f, o)

meta_file_2.close()