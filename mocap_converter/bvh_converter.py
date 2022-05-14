from bvhtomimic import BvhConverter

#inputPath = input("BVH File Path:")
#settingsPath = input("Settings File Path:")
#outputPath = input("Output File Path:")

inputPath = "./walk_sad.bvh"
settingsPath = "settings/bandai_settings.json"
outputPath = "../motion_learning/walk_sad.txt"

converter = BvhConverter(settingsPath)
converter.writeDeepMimicFile(inputPath, outputPath)
