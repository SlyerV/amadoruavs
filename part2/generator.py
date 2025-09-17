# Imports
import os
import random
from PIL import Image, ImageDraw


# Creating Absolute Paths (so that it works wherever you run the code from)
# Creating dataset directory with subfolders (images/train, images/val, labels/train, labels/val)
baseDir = os.path.dirname(os.path.abspath(__file__))
imgDir = os.path.join(baseDir, 'images')
datasetDir = os.path.join(baseDir, "dataset")
dataImgDir = os.path.join(datasetDir, "images")
labelDir = os.path.join(datasetDir, "labels")

os.makedirs(datasetDir, exist_ok=True)
os.makedirs(dataImgDir, exist_ok=True)
os.makedirs(labelDir, exist_ok=True)

for fType in ["train", "val"]:
    os.makedirs(os.path.join(dataImgDir, fType), exist_ok=True)
    os.makedirs(os.path.join(labelDir, fType), exist_ok=True)

# Vars
trainProb = 0.8 # 80:20 ratio for training and validating
sizeMin = 50 # ODLC size range in pixels
sizeMax = 150
blankProb = 0.2 # Probability to not generate ODLC

# Storing Images in List
imgs = []
for file in os.listdir(imgDir):
    imgs.append(file)
imgCount = 0
# Creating ODLC Images/Labels
for img in imgs:
    bg = Image.open(os.path.join(imgDir, img)).convert('RGB')
    img = bg.copy()
    labelList = []
    if random.random() > blankProb:  
        # Generating ODLC
        size = random.randint(sizeMin, sizeMax)
        odlc = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(odlc)
        draw.ellipse((0, 0, size, size), fill=(255, 165, 0, 255))  # Orange ellipse
        
        # Place ODLC in random pos
        x = random.randint(0, img.width - size)
        y = random.randint(0, img.height - size)
        img.paste(odlc, (x, y), odlc)

        # YOLO Label
        xCenter = (x + size/2) / img.width
        yCenter = (y + size/2) / img.height
        width = size / img.width # Width compared to entire img
        height = size / img.height # Height compared to entire img
        labelList.append(f"0 {xCenter} {yCenter} {width} {height}") # Standard YOLO format

    # Deciding if image is train/val
    if random.random() < trainProb:
        folderType = 'train' 
    else:
        folderType = 'val'

    # Save img
    imgName = f"img{imgCount:03d}.png" # Adds 2 padded zeroes
    savePath = os.path.join(baseDir, 'dataset/images', folderType, imgName)
    img.save(savePath)
    # Save label
    labelName = f"img{imgCount:03d}.txt"
    labelPath = os.path.join(baseDir, 'dataset/labels', folderType, labelName)
    with open(labelPath, 'w') as file:
        file.write('\n'.join(labelList))

    imgCount+=1
    print("Created img "+str(imgCount))

print(f"Generated {imgCount} images with labels.")
