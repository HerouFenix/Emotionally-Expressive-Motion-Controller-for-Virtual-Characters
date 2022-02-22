from bvhtomimic import BvhConverter
#Formula: (1.0/scale)*2.54/100.0


scale = 0.05

rFile = open('settings/settings_dance.json', 'r')
lines = rFile.readlines()
rFile.close()

while True:
    if(scale > 1.05): break

    scale_str = "%.2f" % round(scale, 2)
    file_name = 'settings/dance_scales/settings_dance_' + scale_str + '.json'
    file = open(file_name, 'w')

    scale_value = (1.0/scale)*2.54/100.0
    lines[28] = '    "scale": ' + str(scale_value) +',\n'

    file.writelines(lines)
    file.close()

    converter = BvhConverter(file_name)
    converter.writeDeepMimicFile("bvh_data/sad_dance.bvh", "deepmimic_data/dance_scales/sad_dance_test" + scale_str + ".txt")

    scale += 0.05
    scale = round(scale,2)
    print(scale)
