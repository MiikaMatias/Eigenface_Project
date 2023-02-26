from math import sqrt
import os

class vector:

    """A data structure that supports basic vector operations
    with either float or int units..\n
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
            if isinstance(element, (float, int)):
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
    def mean(self):
        return sum(self.values)/len(self)

    def append(self, *t):
        """Checks each element in the given tuple one by one.
        If valid (float, int) then appends to the values of the vector."""

        for element in t:
            if isinstance(element, (float, int)):
                self.values.append(element)
                continue
            raise ValueError(f"""Type vector cannot hold
            elements of type {type(element)}""")
        return self

    def __add__(self, __o):
        if isinstance(__o, (int, float)):
            return vector(*list(map(lambda x: x+__o, self.values)))
        elif isinstance(__o, vector):
            return vector(*list(map(lambda x: x[0]+x[1],
                        zip(self.values, __o.values))))
        else:
            raise ValueError(f"""Type vector cannot be added with
            type {type(__o)}""")

    def __sub__(self, __o):
        if isinstance(__o, (int, float)):
            return vector(*list(map(lambda x: x-__o, self.values)))
        elif isinstance(__o, vector):
            return vector(*list(map(lambda x: x[0]-x[1],
                        zip(self.values, __o.values))))
        else:
            raise ValueError(f"""Type vector cannot be deducted with
            type {type(__o)}""")

    def __mul__(self, __o):
        if isinstance(__o, (int, float)):
            return vector(*list(map(lambda x: x*__o, self.values)))
        elif isinstance(__o, vector):
            return vector(*list(map(lambda x: x[0]*x[1],
                        zip(self.values, __o.values))))
        else:
            raise ValueError(f"""Type vector cannot be multiplied with
            type {type(__o)}""")

    def __truediv__(self, __o):
        if isinstance(__o, (int, float)):
            return vector(*list(map(lambda x: x/__o, self.values)))
        elif isinstance(__o, vector):
            return vector(*list(map(lambda x: x[0]/x[1],
                        zip(self.values, __o.values))))
        else:
            raise ValueError(f"""Type vector cannot be divided with
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
