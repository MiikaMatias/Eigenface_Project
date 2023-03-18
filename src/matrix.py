from src.vector import vector
from src.operations import dot,meanvector
import os

class matrix_datastructure:

    """Used to create the matrix for the images."""

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

        return matrix_datastructure(*new_vecs)

    @property
    def mean_vec(self):
        """Provides the mean vector of the matrix."""
        return meanvector(self)
    
    @property
    def flattened(self):
        """Returns a vector version of the matrix."""
        ret = vector()
        for v in self.vectors:
            ret.append(*v)
        return ret
    
    @property
    def covariance_matrix(self):
        """Returns the covariance matrix of this matrix"""

        m = [vec for vec in self.vectors]

        cov_matrix = matrix_datastructure()
        for i in range(len(m)):
            vec = vector()
            for j in range(len(m)):
                cov = dot(vector(*(m[i]-m[i].mean)),vector(*(m[j]-m[j].mean)))/len(m[i])
                vec.append(cov)
            cov_matrix.append(vec)

        return cov_matrix

    @property
    def meandeducted(self):
        """Takes each vector in the matrix, subtracts the mean vector from it,
        and returns the matrix of these vectors"""
        mean_vector = self.mean_vec
        return_matrix = matrix_datastructure()
        for v in self.vectors:
            return_matrix.append(v-mean_vector)

        return return_matrix
       
    def append(self, v):
        """Appends a vector into the tuple."""
        if isinstance(v, vector):
            if len(self.vectors) == 0 or len(v) == len(self.vectors[0]):
                self.vectors.append(v)
                return self
            else:
                raise TypeError(f'length needs to be {len(self.vectors[0])} now {len(v)}')
        else:
            raise TypeError(f'Cannot append a type {type(v)} into a matrix')

    def __add__(self,__o):
        if isinstance(__o, matrix_datastructure):
            retmatrix = matrix_datastructure()
            for i in range(len(self.vectors)):
                retmatrix.append(vector(*(self.vectors[i]+__o.vectors[i])))
            return retmatrix
        elif isinstance(__o, vector):
            retmatrix = matrix_datastructure()
            for i in range(len(self.vectors)):
                retmatrix.append(vector(*(self.vectors[i]+__o)))
            return retmatrix
        elif isinstance(__o, (int,float)):
            retmatrix = matrix_datastructure()
            for i in range(len(self.vectors)):
                retmatrix.append(vector(*(self.vectors[i]+__o)))
            return retmatrix
        else:
            raise ValueError(f"Can only add type matrix to matrix, not {type(__o)}") 

    def __sub__(self,__o):
        if isinstance(__o, matrix_datastructure):
            retmatrix = matrix_datastructure()
            for i in range(len(self.vectors)):
                retmatrix.append(vector(*(self.vectors[i]-__o.vectors[i])))
            return retmatrix
        elif isinstance(__o, vector):
            retmatrix = matrix_datastructure()
            for i in range(len(self.vectors)):
                retmatrix.append(vector(*(self.vectors[i]-__o)))
            return retmatrix
        elif isinstance(__o, (int,float)):
            retmatrix = matrix_datastructure()
            for i in range(len(self.vectors)):
                retmatrix.append(vector(*(self.vectors[i]-__o)))
            return retmatrix
        else:
            raise ValueError(f"Can only deduct type matrix from matrix, not {type(__o)}") 
    
    def __mul__(self, __o):
        if isinstance(__o, matrix_datastructure):
            retmatrix = matrix_datastructure()
            for i in range(len(self.vectors)):
                retmatrix.append(vector(*(self.vectors[i]*__o.vectors[i])))
            return retmatrix
        elif isinstance(__o, (int,float)):
            retmatrix = matrix_datastructure()
            for i in range(len(self.vectors)):
                retmatrix.append(vector(*(self.vectors[i]*__o)))
            return retmatrix
        else:
            raise ValueError(f"Can only multiply type matrix with matrix, not {type(__o)}") 

    def __truediv__(self, __o):
        if isinstance(__o, matrix_datastructure):
            retmatrix = matrix_datastructure()
            for i in range(len(self.vectors)):
                retmatrix.append(vector(*(self.vectors[i]/__o.vectors[i])))
            return retmatrix
        elif isinstance(__o, int):
            retmatrix = matrix_datastructure()
            for i in range(len(self.vectors)):
                retmatrix.append(vector(*(self.vectors[i]/__o)))
            return retmatrix
        else:
            raise ValueError(f"Can only divide type matrix with matrix, not {type(__o)}") 


    def __eq__(self, __o: object) -> bool:
        return self.vectors == __o.vectors

    def __str__(self) -> str:
        ret_str = ''
        for i in range(len(self.vectors[0])):
            for vector in self.vectors:
                if isinstance(vector[i], (int,float)):
                    ret_str += f"{vector[i]:.3f} \t"
                else:
                    ret_str += f"{vector[i]} \t"
            ret_str += '\n'
        return ret_str

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, item: str):
        return self.vectors[item]