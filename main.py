import data.random_sample as rand
import glob
import numpy as np
import random
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

    # We remove old files ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    eigenfaces_to_remove = glob.glob('data/outputs/eigenfaces/*.jpg')
    for e in eigenfaces_to_remove:
        os.remove(e)

    reconstructions_to_remove = glob.glob('data/outputs/reconstructions/*.jpg')
    for r in reconstructions_to_remove:
        os.remove(r)

    # Now we configure the model –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    
    # use standard data or sample randomly
    print('We can either use standard data, or sample randomly from test_data')
    if input('Would you like to use standard data? (y/n):') != 'n':
        X_train = [[x] for x in sorted(glob.glob('data/train_standard/*.jpg'))]
        y_train = [1,2,3,4,5,6,7,8,9]
        X_test = sorted(glob.glob('data/test_standard/*.jpg'))
        y_test = [1,1,1,2,3,4,5,6,7,8,9]
    else:
        print('\nWe would like to sample some percentage of the images for testing \nRecommended value 0.25\nAny values larger than 1 or smaller than 0 will be replaced with 0.25')
        sample_percentage = float(input('Insert a float between 0-1: '))

        X_train,X_test,y_train,y_test = rand.train_test_split(float(sample_percentage))
    
    # Potentially mitigate lighting by removing some eigenvectors 
    print('\n If we want, we can remove some eigenvectors (1-3).')
    print('The primary reason for doing this is mitigating the effects of light in recognition.')
    remove_vectors = input('would you like to remove eigenvectors (y|n):')
    how_many_to_remove = 0
    if remove_vectors == 'y':
        how_many_to_remove = int(input('how many: '))

    # Set the parameter for PCA
    print('\nNow need parameter € for PCA; \n recommended 0.8 for performance, 0.95 for accuracy')
    variance_treshold = float(input('Give a variance treshold between 0-1: '))

    save_eigen = input('Would you like to save generated eigenfaces for inspection into data/outputs/eigenfaces (y/n): ') == 'y'
    save_ordered = input('Would you like to save faces in order of their accounted variance into data/outputs/sorted_images (y/n): ') == 'y'

    # Configuration over, start calculations –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––– 

    # Form matrix from the images
    sample = [im for im in X_train for im in im]

    print('We use the following images for training:')
    print(*X_train,sep='\n')

    matrix = m()
    
    for f in sample:
        print('vectorizing:',f,end='            \r')
        im = image_to_vec(f)
        matrix.append(im)

    print('\n\n Now we calculate the covariance matrix for the images...')
    
    # We use custom matrix operation to normalize the faces, and then get the eigenvectors of the covariance matrix

    normalized_faces = matrix.meandeducted

    cov_matrix = normalized_faces.covariance_matrix/len(sample)
    print('\nNext we use numpy to calculate eigenvectors for the covariance matrix...')        
    eigenvalues, eigenvectors, = np.linalg.eig(np.array([arr for arr in cov_matrix]))

    # Here we zip y_train with the eigenvectors in order to have the labels for the data

    zipped_eigenvectors = zip(eigenvalues,[v(*list(ev)) for ev in eigenvectors],matrix.vectors)

    sorted_by_eigenvalues = sorted(zipped_eigenvectors, key = lambda x: x[0],reverse=True)

    # Now we have most prominent eigenvectors, let's separate them
    print('\nSorting the vectors...')

    sorted_eigenvalues = [s[0] for s in sorted_by_eigenvalues]
    sorted_eigenvectors = [s[1] for s in sorted_by_eigenvalues]
    sorted_images = [s[2] for s in sorted_by_eigenvalues]

    if save_ordered:
        for i,vector in enumerate(sorted_images):
            vec_to_image(vector).save(f'data/outputs/sorted_images/{i}.jpg')

    print(f'Vectors have been sorted. Now we use the given variance value of {variance_treshold} to get n eigenvectors.')

    # Here we use PCA to derive the amount of necessary eigenvectors based on a treshold given by the user

    k = get_k(sorted_eigenvalues,variance_treshold)
    sorted_eigenvectors = sorted_eigenvectors[:k]

    # Form a matrix by relevance of each eigenvector derived from the size of the eigenvalues

    print('\n We have sorted the eigenvectors. Now we form an eigenmatrix...')

    eigenmatrix = dot(matrix.T,m(*sorted_eigenvectors)).T
    eigenmatrix.vectors = eigenmatrix.vectors[how_many_to_remove:]

    # Save eigenfaces
    if save_eigen:
        for i,eigenface in enumerate(eigenmatrix):
            vec_to_image(eigenface).save(f'data/outputs/eigenfaces/eigenface_{i}.jpg')
    print('Eigenmatrix formed')

    print('\n Deriving weights for each image by projecting it into the eigenmatrix...')

    weights = m(*[dot(eigenmatrix.T, normalized_face) for normalized_face in normalized_faces])
    

    # we can recognize new images now

    print('Now we test accuracy for the testing data as well as any images in data/unknown_images')

    new_images = X_test + sorted(glob.glob('data/unknown_images/*.jpg'))
    i = 0
    while len(y_test) != len(new_images):
        y_test.append('not_a_face') 
        i+=1
    respective_distances = []

    mean = matrix.mean_vec
    
    print('\nWe start recognizing images...')
    for i in range(len(new_images)):
        testfor = image_to_vec(new_images[i])
        normalized = testfor - mean
        new_weights = dot(eigenmatrix.T, normalized)
        euclidean_distance = weights - new_weights
        smallest_dist = min([(y_train[j],vec.mag,new_images[i]) for j,vec in enumerate(euclidean_distance)],key=lambda x: x[1])
        # convert to scientific notation for clarity
        respective_distances.append((smallest_dist[0],f'{smallest_dist[1]:e}',smallest_dist[2]))

    correct = 0
    print('The following are the predictions made by the model:')
    for i,distance in enumerate(respective_distances):
        rough_val = distance[1].split('e+')
        distance_from_origin = float(rough_val[0])*(10**int(rough_val[1]))
        # this distance treshold of 1.5e+08 was chosen with pure, raw emotion
        if distance_from_origin > (2*(10**8)):
            distance = ('not_a_face',distance[1],distance[2])
        print(distance,'correct' if distance[0] == y_test[i] else 'incorrect')
        if distance[0] == y_test[i]:
            correct += 1
    print(f'The model succeeded {correct/len(respective_distances)*100:.3f}% of the time')
    print(f'eigenfaces used: {len(eigenmatrix)}')
    print('Results may get better or worse depending on parameters you have given, and the sample size used!')


def sample_n():
    """Take a sample from test data."""
    n = int(input(f"Please input the amount of each face you'd like to sample from data/test_data/ (4-7): "))
    rand.choose_remove(n,'test_data/')
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

def run_tests():
    """Special testing function for ease of use purposes."""
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

