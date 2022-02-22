# Using readlines()
rFile = open('bvh_data/happy_dance.bvh', 'r')
wFile = open('bvh_data/happy_dance_reduced.bvh', 'w')

lines = rFile.readlines()
newLines = []
indexCounter = 0
count = 0
frameCounter = 0

frameCounterIndex = 0

motionFound = False
addingFrames = False
lastLine = ""

for line in lines:
    
    if(not motionFound):
        if line != "MOTION\n":
            newLines.append(line)
            indexCounter += 1
        else:
            motionFound = True
            newLines.append(line)
            indexCounter += 1
    else:
        
        if(not addingFrames):
            count+=1
            if(count == 1):
                newLines.append(line)
                frameCounterIndex = indexCounter
                indexCounter += 1
            else:
                newLines.append(line)
                indexCounter += 1

                addingFrames = True
                count = 0

        else:
            if(count%5 == 0):
                lastLine = line
                frameCounter+=1
                newLines.append(line)
                indexCounter += 1
            else:
                newLines.append(lastLine)
            count += 1
        
#newLines[frameCounterIndex] = "Frames:	" + str(frameCounter) + "\n"

wFile.writelines(newLines)