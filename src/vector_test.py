import unittest
from src.vector import vector as v

class TestVec(unittest.TestCase):  

    @classmethod
    def setUp(self):
        self.v1 = v(1,2,3)
        self.v2 = v(1,2,3)

    def test_addition(self):
        self.assertEqual(self.v1+self.v2, v(2,4,6))
        self.assertEqual(self.v1+1, v(2,3,4))

    def test_subtract(self):
        self.assertEqual(self.v1-self.v2, v(0,0,0))
        self.assertEqual(self.v1-1, v(0,1,2))

    def test_multiply(self):
        self.assertEqual(self.v1*self.v2, v(1,4,9))
        self.assertEqual(self.v1*2, v(2,4,6))

    def test_truediv(self):
        self.assertEqual(self.v1/self.v2, v(1,1,1))
        self.assertEqual(self.v1/2, v(0.5,1,1.5))
    
    def test_len(self):
        self.assertEqual(len(self.v1), 3)
    
    def test_append(self):
        self.assertEqual(v(1,2,3,1), self.v1.append(1))

    def test_mean(self):
        self.assertEqual(self.v1.mean, 2)

    def test_mag(self):
        self.assertEqual(self.v1.mag, 3.7416573867739413)

    def test_get(self):
        self.assertEqual(self.v1[0],1)

if __name__ == '__main__':
    unittest.main()