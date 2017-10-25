import math
import random

import numpy as np

EPS = 1e-6


def norm(a, b):
    """
    Computes a norm of two numbers. For single numbers, faster than np.linalg.norm
    :param a: float a
    :param b: float b
    :return: norm of (a,b)
    """
    return math.sqrt(a * a + b * b)


class Coordinate(object):
    """
    Class that models a coordinate in Carthesian space.
    Used as a base class for points, velocities and sizes.
    Implemented some basic 'magic' methods to facilitate the use of basic operators
    """

    def __init__(self, x):
        """
        :param x: iterable of coordinates. Requires a list of length 2.
        """
        self.array = np.asarray(x, dtype='float64')
        self.type = self.__class__.__name__

    angle = property(lambda s: math.atan2(s[1], s[0]))

    def __len__(self):
        return len(self.array)

    def __iter__(self):
        return iter(self.array)

    def __getitem__(self, i):
        return self.array[i]

    def __add__(self, other):
        return Point(self.array + other.array)

    def __sub__(self, other):
        return Point(self.array - other.array)

    def __mul__(self, other):
        # Change to self.__class__
        return Point(self.array * other)

    def __truediv__(self, other):
        return Point(self.array / other.array)

    def __repr__(self):
        return "%s(%s)" % (self.type, ", ".join("%.2f" % f for f in self.array))

    def is_zero(self):
        """
        Check whether coordinates are within tolerance of zero point
        :return: True if 2-norm of coordinate is smaller than epsilon
        """
        return norm(self.array[0], self.array[1]) < EPS


class Size(Coordinate):
    """
    Class that models a size. Sizes are an extension of coordinates that cannot be negative.
    """

    def __init__(self, x):
        super().__init__(x)
        if any(self.array < 0):
            raise ValueError("Negative size specified")

    def random_internal_point(self):
        """
        Provides a random internal point within the size
        :return: Point with positive coordinates both smaller than size
        """

        return Point(np.array([random.random() * dim for dim in self.array]))


class Point(Coordinate):
    """
    Class that models a point within a plane.
    """
    x = property(lambda s: s[0])
    y = property(lambda s: s[1])
