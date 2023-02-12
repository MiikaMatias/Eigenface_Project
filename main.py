import logic.lalgebra as l
import data.random_sample as rand
import glob
from PIL import Image
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

"""
At the moment runs exploratory code, will be expanded into a broader program shortly.
Right now can be ran, results are seen in 'data/outputs' as an average face

-test_images_to_average(_grey): these functions calculate an average image from a set
of given images. They're the result of experimenting with the custom matrix datastructure:
data/outputs is where the file is unloaded in either colour or greyscale

-eigenvectors(): calculates the eigenfaces for the given set of n images sampled
from a specific directory; unfinished
"""

def test_images_to_average():

    """
    Outputs average image of the given sample 
    to data/images. Colourized. 
    """

    files = glob.glob('data/images/*.jpg')
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

    files = glob.glob('data/images/*.jpg')
    matrix = l.matrix()
    for f in files:
        im = l.image_to_vec(f,grey=True)
        matrix.append(im)
    im = l.vec_to_image(matrix.means)
    im.save('data/outputs/output.jpg')

def eigenvectors():
    None

def sample_n():
    n = int(input(f"Please input the amount of images you'd like to sample from data/test_data/: "))
    rand.choose_remove(n,'data/test_data/')

if __name__ == '__main__':
    # Note that running the program removes all prior files from the address
    # n will be the amount of images sampled from address
    sample_n()

    print(f"""Instructions:
    1: compute the average image from the sample
    2: compute the eigenvectors for the sample
    3: get a new sample of n from data/test_data
    0: quit""")
    while True:
        cmd = int(input("Command:"))
        if cmd == 1:
            test_images_to_average_grey()
        if cmd == 2:
            eigenvectors()
        if cmd == 3:
            sample_n()
        if cmd == 0:
            break

    print('Done')