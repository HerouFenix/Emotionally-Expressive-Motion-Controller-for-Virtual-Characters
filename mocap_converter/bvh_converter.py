from bvhtomimic import BvhConverter

inputPath = input("BVH File Path:")
settingsPath = input("Settings File Path:")
outputPath = input("Output File Path:")

converter = BvhConverter(settingsPath)
converter.writeDeepMimicFile(inputPath, outputPath)
