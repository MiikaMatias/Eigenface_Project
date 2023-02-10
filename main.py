import logic.lalgebra as l
import data.random_sample as rand
import glob
from PIL import Image
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

"""
At the moment runs exploratory code, will be expanded into a broader program shortly.
Right now can be ran, results are seen in 'data/outputs' as an average face
"""

def test_images_to_average():

    """
    Outputs average image of the given sample 
    to data/images. Colourized. 
    """

    files = glob.glob('data/images/*')
    matrix = l.matrix()
    for f in files:
        im = l.image_to_vec(f)
        matrix.append(im)
    im = l.vec_to_image(matrix.means)
    im.save('data/outputs/output.jpg')

def test_images_to_average_grey():

    """
    Outputs average image of the given sample 
    to data/images. Greyscale. 
    """

    files = glob.glob('data/images*')
    matrix = l.matrix()
    for f in files:
        im = l.image_to_vec(f,grey=True)
        matrix.append(im)
    im = l.vec_to_image(matrix.means)
    im.save('data/outputs/output.jpg')

if __name__ == '__main__':
    # This will be the directory from which images will be randomly sample according to n
    address = 'data/test_data/'
    # Note that running the program removes all prior files from the address

    # n will be the amount of images sampled from address
    n = int(input(f"Please input the amount of images you'd like to sample from '{address}': "))

    rand.choose_remove(n,address)
    test_images_to_average()

    print('Done')