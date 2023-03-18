from src.matrix import matrix_datastructure as m
from src.vector import vector as v
from src.operations import image_to_vec
from PIL import Image
import numpy as np
import unittest


class TestMat(unittest.TestCase):  

    @classmethod
    def setUp(self):
        self.v1 = v(1,1,1)
        v2 = v(2,2,2)
        v3 = v(3,3,3)
        self.m1 = m(self.v1,v2,v3)

        v4 = v(1,2,3)
        v5 = v(1,2,3)
        v6 = v(1,2,3)
        self.m2 = m(v4,v5,v6)


    def test_addition(self):
        self.assertEqual(self.m1+self.m2, m(
            v(2,3,4),
            v(3,4,5),
            v(4,5,6)))
        self.assertEqual(self.m1+1, m(
            v(2,2,2),
            v(3,3,3),
            v(4,4,4)))
        self.assertEqual(self.m1 + self.v1, m(
        v(2,2,2),
        v(3,3,3),
        v(4,4,4) 
        ))

    def test_subtract(self):
        self.assertEqual(self.m1-self.m2, m(
            v(0,-1,-2),
            v(1,0,-1),
            v(2,1,0)))
        self.assertEqual(self.m1-1, m(
            v(0,0,0),
            v(1,1,1),
            v(2,2,2)))
        self.assertEqual(self.m1 - self.v1, m(
        v(0,0,0),
        v(1,1,1),
        v(2,2,2) 
        ))

    def test_multiply(self):
        self.assertEqual(self.m1*self.m2, m(
            v(1,2,3),
            v(2,4,6),
            v(3,6,9)))
        self.assertEqual(self.m1*2, m(
            v(2,2,2),
            v(4,4,4),
            v(6,6,6)))

    def test_truediv(self):
        self.assertEqual(self.m1/self.m2, m(
            v(1.0, 0.5, 0.3333333333333333),
            v(2.0, 1.0, 0.6666666666666666),
            v(3.0, 1.5, 1.0)))
        self.assertEqual(self.m1/2, m(
            v(0.5,0.5,0.5),
            v(1,1,1),
            v(1.5,1.5,1.5)))
    
    def test_len(self):
        self.assertEqual(len(self.m1), 3)

    
    def test_get(self):
        self.assertEqual(self.m1[0], v(1,1,1))
        self.assertEqual(self.m1[0][0], 1)


    def test_eq(self):
        self.assertEqual(self.m1==self.m1, True)
        self.assertEqual(self.m1==self.m2, False)


    def test_append(self):
        self.m2.append(v(1,2,3))
        self.assertEqual(self.m2,
        m(v(1,2,3),
          v(1,2,3),
          v(1,2,3),
          v(1,2,3)))


    def test_covariance_matrix(self):
        images = ['src/test_images/26_0_1_20170117200127227.jpg.chip.jpg',
                  'src/test_images/26_1_1_20170116231925419.jpg.chip.jpg',
                  'src/test_images/26_1_2_20170116175920746.jpg.chip.jpg',
                  'src/test_images/29_0_0_20170104201134466.jpg.chip.jpg',
                  'src/test_images/30_0_0_20170117181207964.jpg.chip.jpg']
        matrix = m(*[image_to_vec(im) for im in images])
        test_cov = matrix.covariance_matrix
        correct_cov = np.cov(np.array([vector.values for vector in matrix]))
        
        for i in range(len(test_cov)):
            for j in range(len(test_cov[0])):
                self.assertEqual(int(test_cov[i][j]), int(correct_cov[i][j]))


    def test_meandeduction(self):
        md = self.m1.meandeducted
        self.assertEqual(md, m(v(-1,-1,-1),
                               v(0,0,0),
                               v(1,1,1)))
        
    
    def test_flattened(self):
        self.assertEqual(self.m1.flattened, v(1,1,1,2,2,2,3,3,3))


    def test_transpose(self):
        self.assertEqual(self.m1.T, self.m2)


    def test_str(self):
        self.assertEqual(str(self.m1), str(self.m1))

if __name__ == '__main__':
    unittest.main()