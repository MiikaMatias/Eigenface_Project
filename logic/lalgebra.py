"""Contains all classes, functions that deal with vectors and matrices,
 and their conversions to images."""
from math import sqrt
from PIL import Image
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def image_to_vec(path_to_pic: str, grey=False):

    """Processes an image and turns it into a grayscale vector."""

    input_image = Image.open(path_to_pic)
    pixel_map = input_image.load()
    width, height = input_image.size
    for i in range(width):
        for j in range(height):
            r, g, b = input_image.getpixel((i, j))
            # greyscale
            if grey:
                grayscale = (0.299*r + 0.587*g + 0.114*b)
                pixel_map[i, j] = (int(grayscale), int(grayscale),
                                   int(grayscale))
            # normal colour using the rbg type
            else:
                pixel_map[i, j] = (int(r), int(g), int(b))

    if grey:
        return vector(*list(map(lambda x: x[0], list(input_image.getdata()))))
    else:
        return vector(*list(map(lambda x: rgb(*x),
                      list(input_image.getdata()))))


def vec_to_image(vec):

    """Processes an image vector (type vector;
    greyscale|rgb) and turns it into an image. Image is always
    precisely square!"""

    if isinstance(vec[0], rgb):
        # This is especially slow due to the fact that we
        # have to map over the image an additional time
        # to convert it into a list of tuples
        # instead of a vector of rgb instances
        im = Image.new('RGB', (int(sqrt(len(vec))), int(sqrt(len(vec)))))
        tuple_list = list(map(lambda x: tuple(map(
            lambda x: int(x), x.val)), vec.values))
        im.putdata(tuple_list)
    else:
        im = Image.new('L', (int(sqrt(len(vec))), int(sqrt(len(vec)))))
        im.putdata(vec.values)

    return im


def meanvector(*vecs):

    """Calculates the row means for a vector;
    can intake either rgb vectors or int/float vectors."""

    if vecs == []:
        return []
    elif isinstance(vecs[0][0], rgb):
        # this portion calculates the mean for rgb valued vectors
        pairs = list(zip(*vecs))
        returnvec = vector()
        for pair in pairs:
            summed = rgb(0, 0, 0)
            for rgb_tuple in pair:
                summed = summed + rgb_tuple
            returnvec.append(summed/(len(pair)))
        return returnvec
    elif isinstance(vecs[0][0], (float,int)):
        return vector(*list(map(lambda x: sum(x)/len(x), list(zip(*vecs)))))
    else:
        raise TypeError("Vector types don't match",vecs)

def dot(vec1, vec2):
    if isinstance(vec1[0], (int,float)) and isinstance(vec1[0],(int,float)):
        ret = list(map(lambda x: x[0]*x[1], list(zip(vec1,vec2))))
        return sum(ret)
    elif isinstance(vec1[0], rgb) and isinstance(vec1[0],rgb):
        ret = list(map(lambda x: x[0]*x[1], list(zip(vec1,vec2))))
        return sum(ret, start=rgb(0,0,0))
    raise TypeError(f'{type(vec1)} and {type(vec2)} are inappropriate')

def covariance_matrix(m):
    """Takes in a matrix and returns a square covariance
    matrix. Diagonal columns are the variance, the rest are
    covariance values."""

    # formula and meaning of covariance are required to
    # understand this code
    
    n = len(m)
    
    ret = matrix()
    for i in range(len(m)):
        current_vec = vector()
        for j in range(len(m.T)):
            mean_1 = sum(list(m[i]))/len(m[i])
            vec_1 = list(m[i] - mean_1)

            mean_2 = sum(list(m[j]))/len(m[j])
            vec_2 = list(m[j] - mean_2)

            current_vec.append(sum(list(map(lambda x: x[0] * x[1], list(zip(vec_1,vec_2)))))/(n-1))

        ret.append(current_vec)
    return ret

class rgb:
    """This is a custom data-structure that is a 3-tuple.
    It has all required operations that are required
    for processing coloured images. It should be noted
    that coloured images are much more performance heavy
    than normal greyscale images."""

    def __init__(self, r, g, b) -> None:
        self.val = (r, g, b)

    def __add__(self, __o):
        if isinstance(__o, rgb) or isinstance(__o, tuple):
            self.val = (self.val[0] + __o.val[0],
                        self.val[1] + __o.val[1],
                        self.val[2] + __o.val[2])
        elif isinstance(__o, int) or isinstance(__o, float):
            self.val = (self.val[0]+__o, 
                        self.val[1]+__o, 
                        self.val[2]+__o)
        else:
            raise TypeError(f'Not a valid type of {type(__o)} for addition with rgb')
        return self

    def __sub__(self, __o):
        if isinstance(__o, rgb) or isinstance(__o, tuple):
            self.val = (self.val[0] - __o.val[0],
                        self.val[1] - __o.val[1],
                        self.val[2] - __o.val[2])
        elif isinstance(__o, int) or isinstance(__o, float):
            self.val = (self.val[0]-__o, self.val[1]-__o, self.val[2]-__o)
        else:
            raise TypeError(f"""Not a valid type of {type(__o)}
            for subtraction from rgb""")
        return self

    def __mul__(self, __o):
        if isinstance(__o, rgb) or isinstance(__o, tuple):
            self.val = (self.val[0] * __o.val[0],
                        self.val[1] * __o.val[1],
                        self.val[2] * __o.val[2])
        elif isinstance(__o, int) or isinstance(__o, float):
            self.val = (self.val[0]*__o, self.val[1]*__o, self.val[2]*__o)
        else:
            raise TypeError(f"""Not a valid type of {type(__o)}
            for division from rgb""")
        return self

    def __truediv__(self, __o):
        if isinstance(__o, rgb) or isinstance(__o, tuple):
            self.val = (self.val[0] / __o.val[0],
                        self.val[1] / __o.val[1],
                        self.val[2] / __o.val[2])
        elif isinstance(__o, int) or isinstance(__o, float):
            self.val = (self.val[0]/__o, self.val[1]/__o, self.val[2]/__o)
        else:
            raise TypeError(f"""Not a valid type of {type(__o)}
            for division from rgb""")
        return self

    def __repr__(self) -> str:
        return str(self.val)

    def __iter__(self):
        return iter(self.val)

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, (tuple,rgb,list)) and len(self.val) == len(__o.val):
            for i in range(len(__o)):
                if self.val[i] != __o[i]:
                    return False
            return True
        else:
            return False

    def __len__(self):
        return len(self.val)

    def __getitem__(self,item):
        return self.val[item]

class vector:

    """A data structure that supports basic vector operations
    with either float/int units or custom rgb 3-tuples.\n
       -addition, subtraction with vectors, real numbers\n
       -multiplication with vectors, real numbers\n
       -magnitude, length, ssq\n
       -subscripting
    """

    def __init__(self, *t: float):
        # Sets the contents of the vector into self.values.
        self.values = []
        for element in t:
            # bool is specified due to Python's
            # dynamic typing interpreting bools as ints.
            if isinstance(element, bool):
                raise ValueError(f"""Type vector cannot hold type
                elements of type {type(element)}""")
            # notice that rgb is an accepted value
            if isinstance(element, (float, int, rgb)):
                self.values.append(element)
                continue
            raise ValueError(f"""Type vector cannot hold type elements of type
            {type(element)}""")

    @property
    def ssq(self):
        """Returns the sum of squares of the vector."""

        return sum(list(map(lambda x: x**2, self.values)))

    @property
    def mag(self):
        """Returns the magnitude, or the 'physical length' of the vector."""

        return sqrt(self.ssq)

    @property
    def length(self):
        """Returns the amount of elements in the vector."""

        return len(self.values)

    def append(self, *t):
        """Checks each element in the given tuple one by one.
        If valid (float, int, rgb) then appends to the values of the vector."""

        for element in t:
            if isinstance(element, bool):
                raise ValueError(f"""Type vector cannot hold
                elements of type {type(element)}""")
            if isinstance(element, (float, int, rgb)):
                self.values.append(element)
                continue
            raise ValueError(f"""Type vector cannot hold
            elements of type {type(element)}""")

    def __add__(self, __o):
        if isinstance(__o, (int, float)):
            return list(map(lambda x: x+__o, self.values))
        elif isinstance(__o, vector):
            return list(map(lambda x: x[0]+x[1],
                        zip(self.values, __o.values)))
        else:
            raise ValueError(f"""Type vector cannot be added with
            type {type(__o)}""")

    def __sub__(self, __o):
        if isinstance(__o, (int, float)):
            return list(map(lambda x: x-__o, self.values))
        elif isinstance(__o, vector):
            return list(map(lambda x: x[0]-x[1],
                        zip(self.values, __o.values)))
        else:
            raise ValueError(f"""Type vector cannot be deducted with
            type {type(__o)}""")

    def __mul__(self, __o):
        if isinstance(__o, (int, float)):
            return list(map(lambda x: x*__o, self.values))
        elif isinstance(__o, vector):
            return list(map(lambda x: x[0]*x[1],
                        zip(self.values, __o.values)))
        else:
            raise ValueError(f"""Type vector cannot be multiplied with
            type {type(__o)}""")

    def __len__(self):
        return len(self.values)

    def __eq__(self, __o):
        return self.values == __o.values

    def __str__(self) -> str:
        return str(self.values)

    def __repr__(self) -> str:
        return str(self.values)

    def __getitem__(self, item):
        return self.values[item]


class matrix:

    def __init__(self, *vecs) -> None:
        self.vectors = []
        for v in vecs:
            if isinstance(v, vector):
                self.vectors.append(v)
                if len(self.vectors) >= 2 and (len(self.vectors[-1]) != len(self.vectors[-2])):
                    raise Exception(f"""vectors must be equally long, now at least
                                    \n {self.vectors[-1]}
                                    \n {self.vectors[-2]}
                                    \ndiffer""")
            else:
                raise ValueError(f'matrix can only include vectors, now tried to include {type(v)}')

    @property
    def T(self):
        """Provides the transpose of the matrix."""
        new_vecs = []

        for i in range(len(self.vectors[0])):
            new_vec = vector()
            for j in range(len(self.vectors)):
                new_vec.append(self.vectors[j][i])
            new_vecs.append(new_vec)

        return matrix(*new_vecs)

    @property
    def means(self):
        """Provides the mean vector of the matrix."""
        return meanvector(*self.vectors)

    def append(self, v):
        """Appends a vector into the tuple."""
        if isinstance(v, vector):
            if len(self.vectors) == 0 or len(v) == len(self.vectors[0]):
                self.vectors.append(v)
            else:
                raise TypeError(f'length needs to be {len(self.vectors[0])} now {len(v)}')
        else:
            raise TypeError(f'Cannot append a type {type(v)} into a matrix')

    def __eq__(self, __o: object) -> bool:
        return self.vectors == __o.vectors

    def __str__(self) -> str:
        ret_str = ''
        for i in range(len(self.vectors[0])):
            for vector in self.vectors:
                if isinstance(vector[i], (int,float)):
                    ret_str += f"{vector[i]:.3f}\t"
                else:
                    ret_str += f"{vector[i]}\t"
            ret_str += '\n'
        return ret_str

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, item: str):
        return self.vectors[item]


if __name__ == "__main__":
    v1 = vector(1,2,3)
    v2 = vector(1,3,3)
    v3 = vector(2,2,3)
    m1 = matrix(v1,v2,v3)
    
    covec = covariance_matrix(m1)
    print(covec)