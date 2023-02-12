import unittest
import lalgebra as l
import numpy as np
from PIL import Image, ImageChops, ImageOps
from math import sqrt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class TestCalc(unittest.TestCase):  

    @classmethod
    def setUp(self):
        None

    def test_vectors(self):
        self.v1 = l.vector(1, 2, 3)
        self.v2 = l.vector(-2, -1, -2)
        self.v3 = l.vector(2,-1,1)

        # test basic operations
        self.assertEqual(self.v1 + 1, [2,3,4])
        self.assertEqual(self.v1 - 1, [0,1,2])
        self.assertEqual(self.v1*2, [2, 4, 6])
        self.assertEqual(self.v1*-0.5, [-0.5, -1, -1.5])
        self.assertEqual(self.v1 + self.v1, [2 , 4 , 6])
        self.assertEqual(self.v1*self.v2, [-2, -2, -6])

        # test mean vector
        self.assertEqual(l.meanvector(l.vector(l.rgb(1,1,1), l.rgb(1,1,1), l.rgb(1,1,1)), l.vector(l.rgb(3,3,3),l.rgb(3,3,3),l.rgb(3,3,3))), l.vector(l.rgb(2,2,2),l.rgb(2,2,2),l.rgb(2,2,2)))
        self.assertEqual(l.meanvector(l.vector(1, 1, 1), l.vector(3,3,3)), l.vector(2,2,2))

        # test append

        # test dot
        self.assertEqual(l.dot(l.vector(1,2,3), l.vector(1,2,3)),14)
        self.assertEqual(l.dot(l.vector(l.rgb(1,2,3),l.rgb(1,2,3),l.rgb(1,2,3)), l.vector(l.rgb(2,2,2),l.rgb(2,2,2),l.rgb(2,2,2))),l.rgb(6,12,18))
        self.assertRaises(TypeError,l.dot(l.vector(l.rgb(1,2,3),l.rgb(1,2,3),l.rgb(1,2,3)), l.vector(2,3,2)))

        # test properties
        self.assertEqual(self.v1.ssq, 14)
        self.assertEqual(self.v1.mag, sqrt(14))
        self.assertEqual(self.v1.length, 3)
        self.assertEqual(len(self.v1), 3)

        # test errors with append, type constraints
        with self.assertRaises(ValueError):
            self.vfail = l.vector(True,False,True)
            self.vfail = l.vector('True','False','True')
            self.vfail = l.vector(self.v1,self.v1,self.v1)
            self.v1.append(True)
            self.v1.append(self.v2)

        # test image conversion
        correct_col = Image.open(
            "test_data/26_0_1_20170117200127227.jpg.chip.jpg")
        testable_col = l.image_to_vec(
                        "test_data/26_0_1_20170117200127227.jpg.chip.jpg",False)


        correct_col_ravel = np.array(correct_col).ravel()
        testable_col_ravel = list(np.array(testable_col).ravel())

        # check length
        self.assertEqual(len(correct_col_ravel),len(testable_col_ravel))
        # test the color of each pixel
        for i in range(len(correct_col_ravel)):
            self.assertEqual(correct_col_ravel[i], testable_col_ravel[i])

        # test if images are equal via something called a bounding box (?? :DDD) from the difference of two images
        self.assertEqual(ImageChops.difference(correct_col, l.vec_to_image(l.vector(*testable_col))).getbbox(),None)
    
        # test image conversion for greyscale
        correct_col = Image.open(
            "test_data/26_0_1_20170117200127227.jpg.chip.jpg")
        correct_col = ImageOps.grayscale(correct_col)
        testable_col = l.image_to_vec(
                        "test_data/26_0_1_20170117200127227.jpg.chip.jpg",True)

        correct_col_ravel = np.array(correct_col).ravel()
        testable_col_ravel = list(np.array(testable_col).ravel())

        # check length for greyscale
        self.assertEqual(len(correct_col_ravel),len(testable_col_ravel))
        # test the color of each pixel ; we check if diff between colours between
        # any two pixels is less than 3 due to different weighings 
        # between PIL and my script; for greyscale
        for i in range(len(correct_col_ravel)):
            self.assertEqual(abs(correct_col_ravel[i] - testable_col_ravel[i]) < 2,True)

        # test if images are equal via something called a bounding box (?? :DDD) 
        # from the difference of two images; for greyscale
        # the (0,0,200,200) thing is equivalent to None with greyscale
        self.assertEqual(ImageChops.difference(correct_col, l.vec_to_image(l.vector(*testable_col))).getbbox(),(0,0,200,200))

    def test_rgb(self):
        rgb_1 = l.rgb(20,20,20)
        
        self.assertEqual(rgb_1+1, l.rgb(21,21,21))
        self.assertEqual(rgb_1+rgb_1, l.rgb(42,42,42))
        self.assertEqual(rgb_1*2, l.rgb(84,84,84))
        self.assertEqual(rgb_1/2, l.rgb(42,42,42))
        self.assertEqual(rgb_1/rgb_1, l.rgb(1,1,1))
        self.assertEqual(rgb_1-1, l.rgb(0,0,0))
        self.assertEqual(rgb_1-rgb_1, l.rgb(0,0,0))
        self.assertEqual(l.rgb(1,2,3)*l.rgb(3,2,1), l.rgb(3,4,3))
        self.assertEqual(l.rgb(1,2,3) == l.rgb(1,2,3), True)
        self.assertEqual(l.rgb(1,2,3) == l.rgb(1,1,3), False)
        self.assertEqual(l.rgb(1,2,3) == '1,2,33', False)

        # test errors
        with self.assertRaises(TypeError):
            test = rgb_1 + str('abc')
            test = rgb_1 - str('abc')
            test = rgb_1 * str('abc')
            test = rgb_1 / str('abc')
        
        self.assertEqual(repr(rgb_1), rgb_1.__repr__()) 

    def test_matrices(self):
        v1 = l.vector(1,1,1)
        v2 = l.vector(2,2,2)
        v3 = l.vector(3,3,3)
        m1 = l.matrix(v1,v2,v3)

        v4 = l.vector(1,2,3)
        v5 = l.vector(1,2,3)
        v6 = l.vector(1,2,3)
        m2 = l.matrix(v4,v5,v6)

        # transpose, means, append
        self.assertEqual(m1, m2.T)
        self.assertEqual(m2.means, l.vector(1,2,3))
        m3 = l.matrix(l.vector(3,3,3),l.vector(3,3,3),l.vector(3,3,3))
        m3.append(l.vector(3,3,3))
        self.assertEqual(m3, l.matrix(l.vector(3,3,3),l.vector(3,3,3),l.vector(3,3,3),l.vector(3,3,3)))
        
        # len, getitem
        self.assertEqual(3, len(m2))
        self.assertEqual(l.vector(1,2,3), m2[0])

        # repr
        self.assertEqual(str(m1), m1.__str__())

        # covariances; preset test
        correct_cov = np.cov(np.array([[1,2,3,3,5,6],[1,3,3,2,6,5],[2,2,3,1,5,5]]))
        test_cov = l.covariance_matrix(l.matrix(l.vector(1,2,3,3,5,6),l.vector(1,3,3,2,6,5),l.vector(2,2,3,1,5,5)))
        for m,n in list(zip(list(correct_cov.ravel()),[m for m in test_cov.vectors for m in m])):
            self.assertAlmostEqual(m,n)

        # covariances; test image
        correct_cov = np.array([np.asarray(ImageOps.grayscale(Image.open(
            "test_data/26_1_1_20170116231925419.jpg.chip.jpg"))).ravel(),
                                        np.asarray(ImageOps.grayscale(Image.open(
            "test_data/26_1_2_20170116175920746.jpg.chip.jpg"))).ravel(),
                                        np.asarray(ImageOps.grayscale(Image.open(
            "test_data/30_0_0_20170117181207964.jpg.chip.jpg"))).ravel()])

        test_cov = l.matrix(l.image_to_vec('test_data/26_1_1_20170116231925419.jpg.chip.jpg',True),
                                                l.image_to_vec('test_data/26_1_2_20170116175920746.jpg.chip.jpg',True),
                                                l.image_to_vec('test_data/30_0_0_20170117181207964.jpg.chip.jpg',True))

        for m,n in list(zip(list(correct_cov.ravel()),[m for m in test_cov.vectors for m in m])):
            self.assertEqual(abs(m-n) < 2, True)

if __name__ == "__main__":
    unittest.main()
