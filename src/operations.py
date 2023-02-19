from src.vector import vector as v
from math import sqrt
from PIL import Image
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def image_to_vec(path_to_pic: str):

    """Processes an image and turns it into a grayscale vector."""

    input_image = Image.open(path_to_pic)
    pixel_map = input_image.load()
    width, height = input_image.size
    for i in range(width):
        for j in range(height):
            r, g, b = input_image.getpixel((i, j))
            # greyscale
            grayscale = (0.299*r + 0.587*g + 0.114*b)
            pixel_map[i, j] = (int(grayscale), int(grayscale),
                                int(grayscale))

    return v(*list(map(lambda x: x[0], list(input_image.getdata()))))
    

def vec_to_image(vec):

    """Processes an image vector (type vector;
    greyscale|rgb) and turns it into an image. Image is always
    precisely square!"""

    im = Image.new('L', (int(sqrt(len(vec))), int(sqrt(len(vec)))))
    im.putdata(vec.values)

    return im

def dot(arg1, arg2):
    from src.matrix import matrix_datastructure as m

    if isinstance(arg1[0], (int,float)) and isinstance(arg1[0],(int,float)):
        ret = list(map(lambda x: x[0]*x[1], list(zip(arg1,arg2))))
        return sum(ret)
    elif isinstance(arg1, m):
        ret = v()
        for vector in arg1.T:
            ret.append(dot(vector,arg2))
        return ret
    raise TypeError(f'{type(arg1)} and {type(arg2)} are inappropriately typed, should be vector and vector')

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

