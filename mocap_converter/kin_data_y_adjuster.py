import os
# assign directory
directory = 'bvh_data/kin_emotions'
 
# iterate over directories
for child_directory in os.listdir(directory):
    path_to_child_directory = os.path.join(directory, child_directory)
    if(child_directory == "META"):
        continue

    for filename in os.listdir(path_to_child_directory):
        f = os.path.join(path_to_child_directory, filename)
        if os.path.isfile(f):
            print(filename)
            with open(f, 'r', encoding='utf-8') as file:
                data = file.readlines()
            
            data[3] = "    OFFSET 0.000 -57.00 0.000\n"
            
            with open(f, 'w', encoding='utf-8') as file:
                file.writelines(data)