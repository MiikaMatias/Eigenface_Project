# Implementation
This document intails a description of the project structure, as well as the way in which the algorithm works, and potential improvements to the current project structure.
## Structure
  The structure of the program outside of the main is as follows:
  1) `vector.py`: Contains a generally holistic implementation of a vector class. In practice this is as array with a few special methods needed for the project.
  2) `matrix.py`: Contains a matrix class that is a list of vectors. Most vector operations are called for each element of a matrix.
  3) `operations.py`: Contains operations that are somewhat detached from vector or matrix datastructures.
  4) `random_sample.py`: Resides in `data` and is responsible for sampling as well as splitting data into train and test portions.
## Functionality
This section explains the functionality of the algorithm
### Rough sketch
The idea of the algorithm is to compress data of images to the point where each 200px x 200px image can be represented by just a list of integers! Each integer is 
called a _weight_ and represents the scalar value for a basis vector in an _image space_ created by principal component analysis. 

![illustration_1](https://user-images.githubusercontent.com/100348027/223536400-48880ea8-be11-4909-ae02-4fbd4e681133.jpg)

The idea is to derive a _sufficient_ amount of basis vectors for an _image space_ where each image can be represented by scaling each basis vector by it's respective _weight_.
These basis vectors are called _eigenfaces_.

![illustration_2](https://user-images.githubusercontent.com/100348027/223538000-f84fefb6-aace-48ec-b484-7952b71b60e2.jpg)

The _eigenfaces_ are turned into a matrix, and then used to derive weights for an unknown image. 


The weights of the original images and the new images are then compared in order to find the 
most likely owner of any given face. The smallest difference between weights indicates the image we guess

### In depth explanation

The program will initially get training data and testing data in the form of images. 
Each image is then converted into a vector of elements ranging from 0-255, reflecting the colour of each pixel in the image. 
The vectors are concatenated into a matrix of size _NXL_, where _N_ is the amount of images and _L_ is the length of each vector. We then normalize this matrix, 
which means that we deduct the mean face from each vector. We now do the following operations on this new _NXL_ matrix.

A covariance matrix is calculated for each vector in the matrix, meaning that we receive an _NXN_ matrix. We extract eigenvectors and eigenvalues from this
matrix using a numpy function. The eigenvalues are then used to evaluate the most important principal components. The `operations` module function `get_k` is then used
to derive value _D_ for a list of vectors with _D_ eigenvectors where _D_ is a lot smaller than N. This is then projected onto the original NXL matrix, providing 
us with a DXL eigenmatrix with each vector being an eigenvector, or an eigenface. These can then be output into `data/outputs/eigenfaces´.

After we have formed the eigenmatrix out of the _D_ necessary eigenvectors, we can project all normalized training images onto it. For each projection, we get a list
of length _D_. These are the _weights_ of each image. Let's call this list of image weights _U_, with each weight list being represented by _u_.

Now that we have _U_, we can start recognizing new images. Given a new image, we normalize and project it onto the eigenmatrix of length _D_. We get the weights of the new images.
Let's call this list _W_. In order to recognize any image _w_ in _W_, we deduct from it each _u_ in _U_. The we take the euclidean distance of this deduction. We list 
all of the distances, and take the smallest one. This will be our guess. If the distance is above a certain treshold, we judge that the image is not known. If it is
further still, we judge that it is not a face at all!

## Improvement ideas
Being able to save a model would be a great improvement to the algorithm. More advanced train_test_split function that the user can give specifications to 
would be great too. Even upgrading the model to an implementation of `fisherface` would be neat, since this model gets fooled by lighting really easily. There's a million ways to go.
