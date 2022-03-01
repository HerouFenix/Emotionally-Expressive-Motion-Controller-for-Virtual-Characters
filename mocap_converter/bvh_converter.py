from bvhtomimic import BvhConverter

inputPath = input("BVH File Path:")
settingsPath = input("Settings File Path:")
outputPath = input("Output File Path:")

#inputPath = "bvh_data/kin_test.bvh"
#settingsPath = "settings/kin_settings.json"
#outputPath = "../motion_learning/kin_test.txt"

converter = BvhConverter(settingsPath)
converter.writeDeepMimicFile(inputPath, outputPath)
