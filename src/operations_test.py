from PIL import Image
from src.matrix import matrix_datastructure as m
from src.vector import vector as v
import os
import numpy as np
import unittest
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class TestOp(unittest.TestCase):  

    @classmethod
    def setUp(self):
        None

    def test_image_to_vec(self):
        from src.operations import image_to_vec, vec_to_image
        my_version = image_to_vec(
            'test_data/26_0_1_20170117200127227.jpg.chip.jpg')
        np_version = np.asarray(Image.open('test_data/26_0_1_20170117200127227.jpg.chip.jpg').convert('L')).ravel()
        for i in range(len(my_version)):
            self.assertEqual(abs(my_version[i]-np_version[i]) < 2, True)
        vec_to_image(my_version)

    def test_dot(self):
        from src.operations import dot
        v1 = v(1,2,3)
        v2 = v(1,2,3)
        self.assertEqual(dot(v1,v2), 14)

        m1 = m(v(2,2,2),v(2,2,2),v(2,2,2))
        self.assertEqual(dot(m1,v1),v(12,12,12))

    def test_meanvector(self):
        from src.operations import meanvector
        v1 = v(1,2,3)
        v2 = v(1,2,3)
        v3 = v(1,2,3)
        m1 = m(v1,v2,v3)
        self.assertEqual(meanvector(m1),v1)

if __name__ == '__main__':  
    unittest.main()