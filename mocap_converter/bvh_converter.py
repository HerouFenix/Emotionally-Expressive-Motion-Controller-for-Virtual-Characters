from bvhtomimic import BvhConverter

#inputPath = input("BVH File Path:")
#settingsPath = input("Settings File Path:")
#outputPath = input("Output File Path:")

inputPath = "./F01A0V1.bvh"
settingsPath = "settings/kin_settings.json"
outputPath = "../lmao_wat.txt"

converter = BvhConverter(settingsPath)
converter.writeDeepMimicFile(inputPath, outputPath)
