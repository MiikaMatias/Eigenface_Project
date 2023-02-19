import os, random, shutil, glob
import os

def choose_remove(n, dir_from):
    """remove the contents from images, then choose n new images from 
    target directory and paste into data/images"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    files = glob.glob('images/*.jpg')
    for f in files:
        os.remove(f)
    for _ in range(n):
        shutil.copy(dir_from+random.choice(os.listdir(dir_from)), 'images/')

if __name__ == "__main__":
    choose_remove(5,'test_data/')