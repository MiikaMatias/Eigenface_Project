from src.vector import vector as v
from math import sqrt
from PIL import Image
import os
import time
os.chdir(os.path.dirname(os.path.abspath(__file__)))


"""Contains operations that are better as external functions than internal to any specific datastructure."""


def get_k(eigenvalues, t):

    """Get the necessary k amount of eigenvectors based on a variance
    treshold t, where t is in range [0-1]."""

    if float(t) > 1:
        raise ValueError('treshold must be smaller than 1')
    treshold = float(t)

    sum_l = 0
    vals = []
    sum_e = sum(eigenvalues)
    print(f'\nsearching minimum amount required to explain {treshold*100}% of variance between images...')
    for i,lambda_value in enumerate(eigenvalues):
        sum_l += lambda_value
        vals.append(lambda_value)
        var = sum_l/sum_e
        print('\r' + f'{var} explained at {i}',sep='')
        # For coolness :-D
        time.sleep(0.015)
        if var > treshold:
            print('picked',i,f'eigenvectors out of {len(eigenvalues)}...')
            return i

def image_to_vec(path_to_pic: str):

    """Processes an image and turns it into a grayscale vector."""

    input_image = Image.open(path_to_pic)
    pixel_map = input_image.load()
    width, height = input_image.size
    for i in range(width):
        for j in range(height):
            try:
                r, g, b = input_image.getpixel((i, j))
            except:
                col = input_image.getpixel((i, j))
                pixel_map[i,j] = int(col)
                continue
            # greyscale
            grayscale = (0.299*r + 0.587*g + 0.114*b)
            pixel_map[i, j] = (int(grayscale), int(grayscale),
                                int(grayscale))
    try:
        return v(*list(map(lambda x: x[0], list(input_image.getdata()))))
    except:
        return v(*list(input_image.getdata()))
    

def vec_to_image(vec):

    """Processes an image vector (type vector;
    greyscale|rgb) and turns it into an image. Image is always
    precisely square!"""

    im = Image.new('L', (int(sqrt(len(vec))), int(sqrt(len(vec)))))
    im.putdata(vec.values)

    return im

def dot(arg1, arg2):
    """ Take the dot product of two matrices/vectors
    """
    from src.matrix import matrix_datastructure as m

    if isinstance(arg1, v) and isinstance(arg2,v):
        ret = list(map(lambda x: x[0]*x[1], list(zip(arg1,arg2))))
        return sum(ret)
    elif isinstance(arg1, m) and isinstance(arg2, v):
        ret = v()
        for vector in arg1.T:
            ret.append(dot(vector,arg2))
        return ret
    elif isinstance(arg2, m) and isinstance(arg1, m):
        ret = m()
        arg2 = arg2.T
        for j,vector_1 in enumerate(arg1):
            vec = v()
            for i,vector_2 in enumerate(arg2):
                vec.append(dot(vector_1,vector_2))
            ret.append(vec)
        return ret
    raise TypeError(f'{type(arg1)} and {type(arg2)} are inappropriately typed')

def meanvector(m):

    """Calculates the row means for a for matrix;
    can intake either rgb vectors or int/float vectors."""

    vecs = m.vectors

    if vecs == []:
        return []
    elif isinstance(vecs[0][0], (float,int)):
        return v(*list(map(lambda x: sum(x)/len(x), list(zip(*vecs)))))
    else:
        raise TypeError("Vector types don't match",vecs)
