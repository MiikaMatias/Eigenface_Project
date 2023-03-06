import os, random, shutil, glob
import os
from PIL import Image

def choose_remove(n,where_from):
    """remove the contents from images, then choose n new images from 
    target directory and paste into data/images"""

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    list_of_images = glob.glob(f'{where_from}*.jpg')
    subject_dict = {1:[],
                    2:[],
                    3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],
                    11:[],12:[],13:[],14:[],15:[]}
    for image in list_of_images:
        for i in range(0,10):
            if f"0{i}" in image:
                subject_dict[i].append(image)
            elif f"1{i}" in image:
                subject_dict[10+i].append(image)

    files = glob.glob('images/*.jpg')
    for f in files:
        os.remove(f)
    for i in range(n):
        for key in subject_dict:
            random.shuffle(subject_dict[key])
            im = subject_dict[key][i].split('/')[-1]
            shutil.copy('test_data/'+im, 'images/')

def gif_to_jpg():
    """Data cleaning function for converting gif-format to jpg. Redundant."""
    gifs = glob.glob('data/test_data/*')
    for image in gifs:
        im = Image.open(image)
        newname = image.split('/')
        name = newname[-1].replace('.','_')
        im.save(f'data/test_data/{name}.jpg')
    print(gifs)
    for f in gifs:
        os.remove(f)

def train_test_split(percent):

    try:
        percent = float(percent)
        if percent < 0 or percent > 1:
            percent = 0.25
    except:
        print('Invalid value, picked 0.25')
        percent = 0.25

    """Returns four dictionaries: X_train,X_test,y_train,y_test where the 
    model tests for 0-75% of the images, n being the input for the function.
    Always takes at least one image, so the variance achieved with input is quite small.
    Erroneus values default to 0.25 for convenience."""

    list_of_images = glob.glob('data/images/*.jpg')
    subject_dict = {1:[],
                    2:[],
                    3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],
                    11:[],12:[],13:[],14:[],15:[]}
    for image in list_of_images:
        for i in range(0,10):
            if f"0{i}" in image:
                subject_dict[i].append(image)
            elif f"1{i}" in image:
                subject_dict[10+i].append(image)

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for key in subject_dict:
        random.shuffle(subject_dict[key])
        i = 0

        # We take some percentage of the data as a training sample according to input
        n = 1 if int(len(subject_dict[key])*percent) == 0 else int(len(subject_dict[key])*percent)

        for _ in range(0,n):
            X_test.append(subject_dict[key][i])
            y_test.append(key)
            i += 1
        X_train.append(subject_dict[key][i:])
        y_train += [key for _ in range(i,len(subject_dict[key]))]

    return X_train,X_test,y_train,y_test

if __name__ == "__main__":
    choose_remove(4,'test_data/')
    dict = train_test_split(0.25)
    print(*dict,sep='\n')