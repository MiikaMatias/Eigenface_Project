import os, random, shutil, glob
import os
from PIL import Image

def choose_remove(n, dir_from):
    """remove the contents from images, then choose n new images from 
    target directory and paste into data/images"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    files = glob.glob('images/*.jpg')
    for f in files:
        os.remove(f)
    for _ in range(n):
        shutil.copy(dir_from+random.choice(os.listdir(dir_from)), 'images/')

def gif_to_jpg():
    gifs = glob.glob('data/test_data/*')
    for image in gifs:
        im = Image.open(image)
        newname = image.split('/')
        name = newname[-1].replace('.','_')
        im.save(f'data/test_data/{name}.jpg')
    print(gifs)
    for f in gifs:
        os.remove(f)

if __name__ == "__main__":
    gif_to_jpg()