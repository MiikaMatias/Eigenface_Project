import os, random, shutil, glob


def choose_remove(n, dir_from):
    """remove the contents from images, then choose n new images from 
    target directory and paste into data/images"""
    files = glob.glob('data/images/*.jpg')
    for f in files:
        os.remove(f)
    for _ in range(n):
        shutil.copy(dir_from+random.choice(os.listdir(dir_from)), 'data/images')


if __name__ == "__main__":
    choose_remove(50,'/home/pimiika/Desktop/faces/UTKFace/')