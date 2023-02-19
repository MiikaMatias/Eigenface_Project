import data.random_sample as rand
import glob
import numpy as np
import os
from PIL import Image
from src.vector import vector as v
from src.matrix import matrix_datastructure as m
from src.operations import image_to_vec, vec_to_image, dot

"""
Computes average images or eigenvectors of images
"""

def test_images_to_average_grey():

    """
    Outputs average image of the given sample 
    to data/images. Greyscale. Test function, not relevant
    to the final project.
    """

    # Collects files from data
    sample = glob.glob('data/images/*.jpg')
    # set up an empty matrix that will be filled with image vectors
    matrix = m()
    for f in sample:
        im = image_to_vec(f)
        matrix.append(im)
    mean_vector = matrix.mean_vec

    im = vec_to_image(mean_vector)
    im.save('data/outputs/output.jpg')
    im.show()

def eigenvectors():
    sample = glob.glob('data/images/*.jpg')
    matrix = m()
    
    for f in sample:
        im = image_to_vec(f)
        matrix.append(im)

    cov_matrix = matrix.covariance_matrix
    print('Covariance matrix:\n',cov_matrix)

    # This function will be replaced with a self-made algorithm
    
    eigenvalues, eigenvectors, = np.linalg.eig(np.array([arr for arr in cov_matrix]))

    eigenvectors = [v(*list(eigenvector)) for eigenvector in eigenvectors]
    print('Eigenvectors:\n',eigenvectors)

    eigenfaces = []
    for eigenvector in eigenvectors:
        eigenfaces.append(dot(matrix, eigenvector))
    
    for eigenface in eigenfaces:
        vec_to_image(eigenface).show()

def sample_n():
    n = int(input(f"Please input the amount of images you'd like to sample from data/test_data/: "))
    rand.choose_remove(n,'test_data/')
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

def run_tests():
    import unittest
    import src.vector_test as vt
    import src.matrix_test as mt
    import src.operations_test as ot
    
    runner = unittest.TextTestRunner(verbosity=2)
    print('Testing vectors')
    runner.run(unittest.makeSuite(vt.TestVec))
    print('Testing matrices')
    runner.run(unittest.makeSuite(mt.TestMat))
    print('Testing operations')
    runner.run(unittest.makeSuite(ot.TestOp))
    print()




if __name__ == '__main__':
    # Note that running the program removes all prior files from the address
    # n will be the amount of images sampled from address
    sample_n()

    while True:
        print(f"""Instructions:
        1: compute the average image from the sample (Fun test function; not directly relevant to final project)
        2: compute the eigenfaces from the sample
        3: get a new sample of n from data/test_data 
        4: run tests
        0: quit""")
        cmd = int(input("Command:"))
        if cmd == 1:
            test_images_to_average_grey()
            print('Done, average image: data/outputs/output.jpg')
            print()
        if cmd == 2:
            eigenvectors()
            print()
        if cmd == 3:
            sample_n()
            print('Done')
            print()
        if cmd == 4:
            run_tests()
        if cmd == 0:
            break
        else:
            None

    print('Done')