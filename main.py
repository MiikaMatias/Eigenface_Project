import data.random_sample as rand
import glob
import numpy as np
import os
from PIL import Image
from src.vector import vector as v
from src.matrix import matrix_datastructure as m
from src.operations import image_to_vec, vec_to_image, dot, get_k

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
    address = 'data/images/*.jpg'
    sample = glob.glob(address)
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

    # remove redundant files
    eigenfaces_to_remove = glob.glob('data/outputs/eigenfaces/*.jpg')
    for e in eigenfaces_to_remove:
        os.remove(e)

    reconstructions_to_remove = glob.glob('data/outputs/reconstructions/*.jpg')
    for r in reconstructions_to_remove:
        os.remove(r)

    sample = glob.glob('data/images/*.jpg')
    matrix = m()
    
    for f in sample:
        im = image_to_vec(f)
        matrix.append(im)

    normalized_faces = matrix.meandeducted

    cov_matrix = normalized_faces.covariance_matrix/len(sample)
    print('Covariance matrix:\n',cov_matrix)
    
    eigenvalues, eigenvectors, = np.linalg.eig(np.array([arr for arr in cov_matrix]))

    zipped_eigenvectors = zip(eigenvalues,[v(*list(ev)) for ev in eigenvectors],matrix.vectors)
    print('Eigenvectors:\n',eigenvectors)

    sorted_by_eigenvalues = sorted(zipped_eigenvectors, key = lambda x: x[0],reverse=True)
    sorted_by_eigenvalues = sorted_by_eigenvalues

    # Now we have most prominent eigenvectors, let's separate them

    sorted_eigenvectors = [s[1] for s in sorted_by_eigenvalues]
    sorted_eigenvalues = [s[0] for s in sorted_by_eigenvalues]
    sorted_images = [s[2] for s in sorted_by_eigenvalues]
    for i,vector in enumerate(sorted_images):
        vec_to_image(vector).save(f'data/outputs/sorted_images/{i}.jpg')

    k = get_k(sorted_eigenvalues,input('Give a variance treshold between 0-1: '))
    sorted_eigenvectors = sorted_eigenvectors[:k]
    # Form a matrix by relevance
    eigenmatrix = dot(matrix.T,m(*sorted_eigenvectors)).T
    for i,eigenface in enumerate(eigenmatrix):
        vec_to_image(eigenface).save(f'data/outputs/eigenfaces/eigenface_{i}.jpg')

    weights = m(*[dot(eigenmatrix.T, normalized_face) for normalized_face in normalized_faces])
    
    print('\n','weights:')
    for i,w in enumerate(weights):
        print(i+1,w)

    # we recognize new images now

    new_images = sorted(glob.glob('data/unknown_images/*.jpg'))
    respective_distances = []

    mean = matrix.mean_vec
    
    for i in range(len(new_images)):
        testfor = image_to_vec(new_images[i])
        normalized = testfor - mean
        new_weights = dot(eigenmatrix.T, normalized)
        euclidean_distance = weights - new_weights
        smallest_dist = min([(j,vec.mag,new_images[i]) for j,vec in enumerate(euclidean_distance)],key=lambda x: x[1])
        # convert to scientific notation
        respective_distances.append((smallest_dist[0],f'{smallest_dist[1]:e}',smallest_dist[2]))

    print(*respective_distances,sep='\n')


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
        1: Try to recognize faces from 'data/unknown_images/' with treshold variance
        2: compute the average image from the sample (Fun test function; not directly relevant to final project)
        3: get a new sample of n from data/test_data 
        4: run tests
        0: quit""")
        cmd = int(input("Command:"))
        if cmd == 1:
            eigenvectors()
            print()
        if cmd == 2:
            test_images_to_average_grey()
            print('Done, average image: data/outputs/output.jpg')
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