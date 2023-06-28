import time
import numpy as np
from pyIMU.quaternion import Quaternion, Vector3D, DEG2RAD, EPSILON 
import random

if __name__ == '__main__':

    ######################################
    # Functionality Tests
    ######################################

    # Lets assume plane conventions.
    # From pilots point of view:
    #
    # x points forward (North), positive roll turns plane clockwise
    # y points right (East), positive pitch points nose up
    # z points down, positive yaw turns nose right
    # z points down because of z follows right hand rule of x to y
    
    print("Smallest number is: " + str(EPSILON))
    # Initilization
    print("Results should be 1,2,3")
    x1 = Vector3D(1,2,3)
    print(x1)
    x2 = Vector3D(x=1,y=2,z=3)
    print(x2)
    x3 = Vector3D(np.array([1,2,3]))
    print(x3)
    # Addition
    print("Results should be 2,4,6")
    x4 = x1+x2
    print(x4)
    print("Results should be 2,3,4")
    x5 = x1+1
    print(x5)
    # Subtraction
    print("Results should be 0,1,2")
    x6 = x1-1
    print(x6)
    print("Results should be 0,0,0")
    x7 = x1-x2
    print(x7)
    # Multiplication
    print("Results should be 2,4,6")
    x8 = x1*2
    print(x8)
    print("Results should be 1,4,9")
    x9 = x1*x2
    print(x9)
    # Division
    print("Results should be 0.5,1,1.5")
    x10 = x1/2
    print(x10)
    print("Results should be 1,1,1")
    x11 = x1/x2
    print(x11)
    # Potentiation
    print("Results should be 1,4,9")
    x12 = x1**2
    print(x12)
    print("Results should be 1,4,27")
    x13 = x1**x2
    print(x13)
    # Equality
    print("Results should be True")
    print(x1 == x2)
    print("Results should be False")
    print(x1 == x2*2)
    # Absolute
    print("Results should be 1,2,3")
    print(x1.abs())

