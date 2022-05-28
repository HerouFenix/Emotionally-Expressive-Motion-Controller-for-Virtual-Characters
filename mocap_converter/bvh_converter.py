from bvhtomimic import BvhConverter

#inputPath = input("BVH File Path:")
#settingsPath = input("Settings File Path:")
#outputPath = input("Output File Path:")

inputPath = "./bandai_3.bvh"
settingsPath = "settings/bandai_settings.json"
outputPath = "../motion_learning/bandai_3.txt"

converter = BvhConverter(settingsPath)
converter.writeDeepMimicFile(inputPath, outputPath)
