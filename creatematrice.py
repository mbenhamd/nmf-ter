import glob
import os
import sys
from PIL import Image
from resizeimage import resizeimage
import numpy as np
from sklearn.utils import shuffle
from sklearn import preprocessing

size=None # A CHANGER
threshold = 130

if len(sys.argv) > 1 :
	size = int(sys.argv[1])
if len(sys.argv) > 2 :
	threshold = int(sys.argv[2])

folder = "PeopleArt/resized/"+str(size)+"/"


print("Step 1 - Resize ...")

if not os.path.exists(folder):
    os.makedirs(folder)

images = glob.glob("PeopleArt/JPEGImages/*/*")

i = 1
for image in images:
    label = image.split("/")[2].lower()
    with open(image, 'r+b') as f:
        img = Image.open(f)
        img = resizeimage.resize_cover(img, [size, size])
        img.save(folder+label+"_"+str(i)+'.jpg', img.format)
    i += 1



print("Step 2 - Convert to GreyScale ...")

images = shuffle(glob.glob(folder+"*"))

labels = np.array([])
X_data = np.array([])

for image in images:    
    labels = np.append(labels,image.split("/")[3].split("_")[0].lower())
    
    image = Image.open(image).convert('L')
    arr = np.array(image)
    arr = arr.reshape(1,size*size)
    
    if X_data.any() :
        X_data = np.concatenate( (X_data, arr), axis=0  )
    else:
        X_data = arr


le = preprocessing.LabelEncoder()
le.fit(labels)
num_labels = le.transform(labels)

np.savetxt('matrice-greyscale-'+str(size)+'.csv', X_data, fmt='%3d', delimiter=',')
np.savetxt('labels-'+str(size)+'.csv', num_labels, fmt='%3d', delimiter=',')
np.savetxt('annotation-'+str(size)+'.csv', labels, fmt='%3s', delimiter=',')


print("Step 3 - Convert to binary ...")

X_data_bin = np.array([])
for image in images:
    image = Image.open(image).convert('1')
    arr = arr.reshape(1,size*size)
    if X_data_bin.any() :
        X_data_bin = np.concatenate( (X_data_bin, arr), axis=0  )
    else:
        X_data_bin = arr


np.savetxt('matrice-binary-'+str(size)+'.csv', X_data_bin, fmt='%3d', delimiter=',')


print("Step 4 - Numerical matrix resized ...")

X_data_bin = np.array([])
for image in images:
    image = Image.open(image).convert('P')
    arr = arr.reshape(1,size*size)
    if X_data_bin.any() :
        X_data_bin = np.concatenate( (X_data_bin, arr), axis=0  )
    else:
        X_data_bin = arr


np.savetxt('matrice-numerical-'+str(size)+'.csv', X_data_bin, fmt='%3d', delimiter=',')




