import time
from pyIMU.quaternion import Quaternion, Vector3D, DEG2RAD, EPSILON, RAD2DEG, r33toq
from pyIMU.utilities  import invSqrt, accel2rpy, accel2q, q2rpy, rpy2q, q2gravity, sensorAcc, accelmag2q, accelmag2rpy, heading
from scipy.spatial.transform import Rotation
import random
import math
import numpy as np
from copy import copy
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

    # Converting rpy to quaternion and then back to rpy:
    print("Testing rpy2q and q2rpy")
    
    print(" Roll 90, Pitch 0, Yaw 0")
    print("  Following should be 1.5708, 0.0, 0.0")
    print("  " + str(q2rpy(rpy2q(r=90*DEG2RAD,p=0.0,y=0.0))))
    
    print(" Roll 0, Pitch 90, Yaw 0")
    print("  Following should be 0.0, 1.5708, 0.0")
    print("  But its pi, 1.5708, pi which is the same")
    print("  " + str(q2rpy(rpy2q(r=0.0,p=90*DEG2RAD,y=0.0))))    # sort of same 
    
    print(" Roll 0, Pitch 0, Yaw 90")    
    print("  Following should be 0.0, 0.0, 1.5708")
    print("  " + str(q2rpy(rpy2q(r=0.0,p=0.0,y=90*DEG2RAD))))    # pass

    print(" Roll 45, Pitch 45, Yaw 90")    
    print("  Following should be 0.785, 0.785, 1.5708")
    print("  " + str(q2rpy(rpy2q(r=45.*DEG2RAD,p=45.*DEG2RAD,y=90*DEG2RAD))))    # pass
    
    print("Testing gravity")
    g = Vector3D(0,0,1)
    print(" Gravity vector in x=North y=East z=Down:")
    print(" " + str(g))

    # Roll 90 degrees, points y axis downward, 
    # gravity (0,0,1) should now be on y axis 0,1,0
    print(" Rotate system 90 degrees clockwise around X (North)")
    q = rpy2q(r=90*DEG2RAD,p=0.0,y=0.0)
    print("  The following 3 lines should be: 0,1,0")
    print("   q.T * Gravity * q: " + str((q.conjugate * Quaternion(g) * q).v)) # ok
    print("   q2gravity        : " + str(q2gravity(q)))                        # ok
    print("   R33 * gravity    : " + str(g.rotate(q.r33)))                     # ok

    # Roll -90 degrees, points y axis downward, 
    # gravity (0,0,1) should now be on y axis 0,-1,0
    print(" Rotate system 90 degrees counter clockwise around X (North)")
    q = rpy2q(r=-90*DEG2RAD,p=0.0,y=0.0)
    print("  The following 3 lines should be: 0, -1, 0")
    print("   q.T * Gravity * q" + str((q.conjugate * Quaternion(g) * q).v)) # ok
    print("   q2gravity        " + str(q2gravity(q)))                              # ok
    print("   R33 * gravity    " + str(g.rotate(q.r33)))                     # ok

    # Pitch 90 degrees, points x upwards
    # gravity (0,0,1) should now be on x axis backwards -1,0,0
    
    print(" Rotate system 90 degrees clockwise around Y (East), Pitch, points nose up")
    print("  The following 3 lines should be: -1, 0, 0")
    q = rpy2q(r=0.0,p=90*DEG2RAD,y=0.0)
    print("   q.T * Gravity * q" + str((q.conjugate * Quaternion(g) * q).v)) # ok
    print("   q2gravity        " + str(q2gravity(q)))                              # ok
    print("   R33 * Gravity    " + str(g.rotate(q.r33)))                     # ok

    # Pitch 90 degrees, points x upwards
    # gravity (0,0,1) should now be on x axis backwards -1,0,0
    print(" Rotate system 90 degrees counter clockwise around Y (East), Pitch, points nose up")
    print("  The following 3 lines should be: 1, 0, 0")
    q = rpy2q(r=0.0,p=-90*DEG2RAD,y=0.0)
    print("   q.T * Gravity * q" + str((q.conjugate * Quaternion(g) * q).v)) # ok
    print("   q2gravity        " + str(q2gravity(q)))                              # ok
    print("   R33 * Gravity    " + str(g.rotate(q.r33)))                     # ok

    # Yaw 90 degrees, x becomes y and y becomes -x,
    # gravity (0,0,1) should remain same 0,0,1
    print(" Rotate system 90 degrees clockwise around Z (Down), Yaw, nose to right")
    print("  The following 3 lines should be: 0, 0, 1")
    q = rpy2q(r=0.0,p=0.0,y=90*DEG2RAD)
    print("   q.T * Gravity * q" + str((q.conjugate * Quaternion(g) * q).v)) # ok
    print("   q2gravity        " + str(q2gravity(q)))                              # ok
    print("   R33 * Gravity    " + str(g.rotate(q.r33)))                     # ok

    # Yaw 90 degrees, x becomes y and y becomes -x,
    # gravity (0,0,1) should remain same 0,0,1
    print(" Rotate system 90 degrees counter clockwise around Z (Down), Yaw, nose to left")
    print("  The following 3 lines should be: 0, 0, 1")
    q = rpy2q(r=0.0,p=0.0,y=-90*DEG2RAD)
    print("   q.T * Gravity * q" + str((q.conjugate * Quaternion(g) * q).v)) # ok
    print("   q2gravity        " + str(q2gravity(q)))                              # ok
    print("   R33 * Gravity    " + str(g.rotate(q.r33)))                     # ok

    # No rotation
    # gravity (0,0,1) should remain same 0,0,1
    print(" Rotate system 0 degrees")
    print("  The following 3 lines should be: 0, 0, 1")
    q = rpy2q(r=0.0,p=0.0,y=0.0)
    print("   q.T * Gravity * q" + str((q.conjugate * Quaternion(g) * q).v)) # ok
    print("   q2gravity        " + str(q2gravity(q)))                       # ok
    print("   R33 * Gravity    " + str(g.rotate(q.r33)))              # ok

    # Yaw 90 degrees, x becomes y and y becomes -x,
    # Flying east making right turn
    print(" Rotate system 90 degrees clockwise around Z (Down), and 45degrees around X")
    print("  The following 3 lines should be: 0, 0.707, 0.707")
    q = rpy2q(r=45*DEG2RAD,p=0.0,y=90*DEG2RAD)
    print("   q.T * Gravity * q" + str((q.conjugate * Quaternion(g) * q).v)) # ok
    print("   q2gravity        " + str(q2gravity(q)))                        # ok
    print("   R33 * Gravity    " + str(g.rotate(q.r33)))                     # ok
    
    # Converting gravity to rpy should result in 0,0,0 
    # rpy of 0,0,0 converted to quaternion should result in ?
    # and quaternion converted back to rpy should result in 0,0,0
    print("Convert gravity to RPY and back to quaternion and back to RPY")
    print(" This helps to estimate stationary pose if only accelerometer is available")
    print(" This tests accel2q and accel2rpy")
    print(" The following line should be: 0, 0, 0")
    r=accel2rpy(g) # Should result in r=0,p=0,(y=0): ok
    print(" " + str(r))
    print(" The following 2 lines should be: 1, 0, 0, 0")
    q = rpy2q(r=0.0,p=0.0,y=0.0)
    print(" " + str(q))
    print(" which is the same as:" + str(accel2q(g)))            # should be the same as previous print : ok
    print(" The following line should be: 0, 0, 0")
    print(" " + str(q2rpy(q)))                                   # ok
    # Gravity to quaternion should be same as rpy2q of 0,0,0
    print(" The following 2 lines should be: 1, 0, 0, 0")
    print(" " + str(accel2q(g)))                                 # ok
    print(" " + str(rpy2q(r=0.0,p=0.0,y=0.0)))                   # ok

    print("For this test we need to keep yaw a 0 because we can not determine it from accelerometer readings")
    print("Roll 45 and pitch 45, upwards right turn")
    q = rpy2q(r=45.0*DEG2RAD,p=45.0*DEG2RAD,y=0)
    print(" Quaternion is: " + str(q))
    print("   Rotated Gravity should be -0.707, 0.5, 0.5 and is: ")
    print("    q.T * Gravity * q" + str((q.conjugate * Quaternion(g) * q).v)) # ok
    print("    q2gravity        " + str(q2gravity(q)))                        # ok
    print("    R33 * Gravity    " + str(g.rotate(q.r33)))                     # ok
    g_rot = g.rotate(q.r33)

    print(" Rotated Gravity to rpy and rpy to q is: ", rpy2q(accel2rpy(g_rot))) # ok
    print(" This should be the same as:             ", str(q))                  # ok

    print(" Rotated Gravity rotated to q and q2rpy is: ", RAD2DEG*q2rpy(accel2q(g_rot))) # ok
    print(" This should be the same as:                ", 45.0, 45.0, 0)                 # ok
        
    print(" Quaternion from gravity is: " + str(accel2q(g_rot)))              # ok
    print(" Quaternions should be:      " + str(q))                           # ok
    
    print(" RPY from gravity is:"  + str(RAD2DEG*accel2rpy(g_rot)))               # ok
    print(" RPY above should be:`" + str(45.) + ", " + str(45.) + ", " + str(0.)) # ok

    # Repeat the above example with roll=-30 and pitch=60    
    print("For this test we need to keep yaw a 0 because we can not determine it from accelerometer readings")
    print("Roll -30 and pitch 60, upwards left turn")
    q = rpy2q(r=-30.0*DEG2RAD,p=60.0*DEG2RAD,y=0)
    print(" Quaternion is: " + str(q))
    print("   Rotated Gravity should be -0.866, -0.25, 0.43 and is: ")
    print("    q.T * Gravity * q" + str((q.conjugate * Quaternion(g) * q).v)) # ok
    print("    q2gravity        " + str(q2gravity(q)))                        # ok
    print("    R33 * Gravity    " + str(g.rotate(q.r33)))                     # ok
    g_rot = g.rotate(q.r33)

    print("Rotated Gravity to rpy and rpy to q is: ", rpy2q(accel2rpy(g_rot))) # ok
    print("This should be the same as:             ", str(q))                  # ok

    print("Rotated Gravity rotated to q and q2rpy is: ", RAD2DEG*q2rpy(accel2q(g_rot))) # ok
    print("This should be the same as:                ", -30.0, 60.0, 0)                # ok
        
    print(" Quaternion from gravity is: " + str(accel2q(g_rot)))              # ok
    print(" Quaternions should be:      " + str(q))                           # ok
    
    print(" RPY from gravity is:"  + str(RAD2DEG*accel2rpy(g_rot)))               # ok
    print(" RPY above should be:`" + str(-30.) + ", " + str(60.) + ", " + str(0.)) # ok

    # Repeat the above example with roll=-60 and pitch=-30    
    print("For this test we need to keep yaw a 0 because we can not determine it from accelerometer readings")
    print("Roll -60 and pitch -30, upwards left turn")
    q = rpy2q(r=-60.0*DEG2RAD,p=-30.0*DEG2RAD,y=0)
    print(" Quaternion is: " + str(q))
    print("   Rotated Gravity should be 0.5, -0.75, 0.43 and is: ")
    print("    q.T * Gravity * q" + str((q.conjugate * Quaternion(g) * q).v)) # ok
    print("    q2gravity        " + str(q2gravity(q)))                        # ok
    print("    R33 * Gravity    " + str(g.rotate(q.r33)))                     # ok
    g_rot = g.rotate(q.r33)

    print("Rotated Gravity to rpy and rpy to q is: ", rpy2q(accel2rpy(g_rot))) # ok
    print("This should be the same as:             ", str(q))                  # ok

    print("Rotated Gravity rotated to q and q2rpy is: ", RAD2DEG*q2rpy(accel2q(g_rot))) # ok
    print("This should be the same as:                ", -60.0, -30.0, 0)                # ok
        
    print(" Quaternion from gravity is: " + str(accel2q(g_rot)))              # ok
    print(" Quaternions should be:      " + str(q))                           # ok
    
    print(" RPY from gravity is:"  + str(RAD2DEG*accel2rpy(g_rot)))               # ok
    print(" RPY above should be:`" + str(-60.) + ", " + str(-30.) + ", " + str(0.)) # ok

    # Compare this code with online calculators
    # Google online quaternion converter
    # Rotation Matrix is cosine matrix
    # Roll Pitch Yaw    
    
    # Woflram Alpha 
    # RPY (X,Y,Z)          0,0,0                    Wolfram Alpha RPY (1-2-3)
    # Quaternion (w,x,y,z) 1,0,0,0                  quaternions.online, energid, ninja, topo, matlab, this code
    # Quaternion (w,x,y,z) 1,NA,NA,NA               Wolfram Alpha
    # Rotation Matrix      [1,0,0],[0,1,0],[0,0,1]  geschler, ninja, matlab, this code
    # Angle Axis (a,x,y,z)  0,1,0,0                 ninja
    # Angle Axis            0,0,0,0                 energid, geschler  
    
    # RPY (X,Y,Z)          90,0,0                   Wolfram Alpha RPY (1-2-3)
    # Quaternion (w,x,y,z) 0.7,0.7,0,0              Wolfram Alpha, quaternions.online, gaschler, energid, ninja, matlab, this code
    # Rotation Matrix      [1,0,0],[0,0,1],[0,-1,0] Wolfram Alpha, ninja, matlab
    # Rotation Matrix      [1,0,0],[0,0,-1],[0,1,0] gaschler, energid, ninja, this code
    # Angle Axis (a,x,y,z) 90,1,0,0                 gaschler, energid, ninja

    # RPY (X,Y,Z)          0,0,90
    # Quaternion (w,x,y,z) 0.7,0,0,0.7              Wolfram alpha, quaternions.online, gaschler, ninja, matlab, this code
    # Rotation Matrix      [0,-1,0],[1,0,0],[0,0,1] gaschler, energid, ninja, this code
    # Rotation Matrix      [0,1,0],[-1,0,0],[0,0,1] Wolfram Alpha
    # Angle Axis           90,0,0,1                 gaschler,energid, ninja

    # roll, pitch, yaw = 0, 0, 0
    # r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    # quaternion = r.as_quat()
    # print("Scipy quaternion is:")
    # print(quaternion)
    # print("Scipy quaternion is not explained in documentation, so likely representation is x,y,z,w?")

    print("Comparing RPY to Rotation Matrix")
    q=rpy2q(0,0,0)
    print(" The following lines should be: 1,0,0,0")
    print(" " + str(q))
    print(" The following lines should be: [1,0,0],[0,1,0],[0,0,1]")
    print(" " + str(q.r33))
    
    q=rpy2q(90*DEG2RAD,0,0)
    print(" The following lines should be: 0.707, 0.707, 0, 0")
    print(" " + str(q))
    print(" The following lines should be: [1,0,0],[0,0,-1],[0,1,0]")
    print(" " + str(q.r33))
        
    q=rpy2q(0,90*DEG2RAD,0)
    print(" The following lines should be: 0.7, 0, 0.7, 0")
    print(" " + str(q))
    print(" The following lines should be: [0, 0, 1],[0, 1, 0],[-1, 0, 0]")
    print(" " + str(q.r33))
    
    q=rpy2q(0,0,90*DEG2RAD)
    print(" The following lines should be: 0.7, 0, 0, 0.7")
    print(" " + str(q))
    print(" The following lines should be: [0,-1,0],[1,0,0],[0,0,1]")
    print(" " + str(q.r33))

    # LATEST CODE CHECKINGS
     
    print("Quaternion to Rotation Matrix back to Quaternion")
    q=rpy2q(0,0,0)
    print(" The following lines should be: " + str(q))
    print("                                " + str(r33toq(q.r33)))
    
    q=rpy2q(90*DEG2RAD,0,0)
    print(" The following lines should be: " + str(q))
    print("                                " + str(r33toq(q.r33)))

    q=rpy2q(0,90*DEG2RAD,0)
    print(" The following lines should be: " + str(q))
    print("                                " + str(r33toq(q.r33)))

    q=rpy2q(0,0,90*DEG2RAD)
    print(" The following lines should be: " + str(q))
    print("                                " + str(r33toq(q.r33)))

    q=rpy2q(30*DEG2RAD,60*DEG2RAD,0)
    print(" The following lines should be: " + str(q))
    print("                                " + str(r33toq(q.r33)))

    q=rpy2q(-60*DEG2RAD,30*DEG2RAD,0)
    print(" The following lines should be: " + str(q))
    print("                                " + str(r33toq(q.r33)))

    q=rpy2q(-30*DEG2RAD,-60*DEG2RAD,0)
    print(" The following lines should be: " + str(q))
    print("                                " + str(r33toq(q.r33)))

    ######################################
    # AHRS essential functions
    ######################################
    print('===========================================================================================')
    print('=================AHRS======================================================================')
    print('===========================================================================================')
    print('Testing Residuals -30, -60, 0')
    q = rpy2q(-30*DEG2RAD,-60*DEG2RAD,0)
    g = q2gravity(q)
    print("Gravity is: " + str(g))
    print("The following lines should be: 0, 0, 0")
    acc_residuals = sensorAcc(9.78*g, q, 9.78)
    print(" " + str(acc_residuals))

    print('Testing Residuals 60, 30, 0')
    q = rpy2q(60*DEG2RAD,30*DEG2RAD,0)
    g = q2gravity(q)
    print("Gravity is: " + str(g))
    print("The following lines should be: 0, 0, 0")
    acc_residuals = sensorAcc(9.78*g, q, 9.78)
    print(" " + str(acc_residuals))
 
    print('Testing accelmag2q and accelmag2rpy and heading')

    print(' Testing 0, 0, 0') # ok passed
    q = rpy2q(0*DEG2RAD,0*DEG2RAD,0*DEG2RAD) # make plane point straight forward
    mag = Vector3D(1,0,1)      # make magnetic field point forward and down
    g = q2gravity(q)
    print("  Gravity is:                 " + str(g))
    print("  Magnetic field is:          " + str(mag))
    print("  Quaternion should be:       " + str(q))
    print("  Quaternion from acc,mag is: " + str(accelmag2q(g, mag)))
    print("  RPY should be:              " + str(q2rpy(q)*RAD2DEG))
    print("  RPY should be:               0, 0, 0")
    print("  RPY from acc,mag is:        " + str(q2rpy(accelmag2q(g, mag))*RAD2DEG)) 
    print("  RPY from acc,mag is direct: " + str(accelmag2rpy(g, mag)*RAD2DEG))
    print("  Heading is: " + str(heading(q=q, mag=mag, declination=0.0)*RAD2DEG))

    print(' Testing 90, 0, 0')
    q = rpy2q(30*DEG2RAD,60*DEG2RAD,0*DEG2RAD) # make plane point straight forward
    mag = Vector3D(1,0,1)                      # make magnetic field point forward and down
    print("  Unrotated magnetic field is: " + str(mag))
    mag = mag.rotate(q.r33)
    print("  Rotated magnetic field is:   " + str(mag))
    g = Vector3D(0,0,1)
    print("  Gravity is:                  " + str(g))
    g = g.rotate(q.r33)
    print("  Rotated Gravity is:          " + str(g))
    g = q2gravity(q)
    print("  Gravity from q is:           " + str(g))
    print("  Magnetic field is:          " + str(mag))
    print("  Quaternion should be:       " + str(q))
    print("  Quaternion from acc,mag is: " + str(accelmag2q(g, mag)))       # NOT PASSED
    print("  RPY should be:              " + str(q2rpy(q)*RAD2DEG))
    print("  RPY should be:               90, 0, 0")
    print("  RPY from acc,mag is:        " + str(q2rpy(accelmag2q(g, mag))*RAD2DEG)) # this now include accmag2rpy and rpy2q
    print("  RPY from acc,mag is direct: " + str(accelmag2rpy(g, mag)*RAD2DEG))
    print("  Heading is: " + str(heading(q=q, mag=mag, declination=0.0)*RAD2DEG))
 
    print(' Testing 30, 60, 0')
    q = rpy2q(30*DEG2RAD,60*DEG2RAD,0*DEG2RAD) # make plane point straight forward
    mag = Vector3D(1,0,1)                      # make magnetic field point forward and down
    print("  Unrotated magnetic field is: " + str(mag))
    mag = mag.rotate(q.r33)
    print("  Rotated magnetic field is:   " + str(mag))
    g = Vector3D(0,0,1)
    print("  Gravity is:                  " + str(g))
    g = g.rotate(q.r33)
    print("  Rotated Gravity is:          " + str(g))
    g = q2gravity(q)
    print("  Gravity from q is:           " + str(g))
    print("  Magnetic field is:           " + str(mag))
    print("  Quaternion should be:        " + str(q))
    print("  Quaternion from acc,mag is:  " + str(accelmag2q(g, mag))) # passed
    print("  RPY should be:               " + str(q2rpy(q)*RAD2DEG))
    print("  RPY should be:               30, 60, 0")                 # passed
    print("  RPY from acc,mag is:        " + str(q2rpy(accelmag2q(g, mag))*RAD2DEG)) # passed
    print("  RPY from acc,mag is direct: " + str(accelmag2rpy(g, mag)*RAD2DEG))   # passed
    print("  Heading is: " + str(heading(q=q, mag=mag, declination=0.0)*RAD2DEG)) # passed

    print(' Testing 30, 60, -30')
    q = rpy2q(30*DEG2RAD,60*DEG2RAD,-30*DEG2RAD) # make plane point straight forward
    mag = Vector3D(1,0,1)                      # make magnetic field point forward and down
    print("  Unrotated magnetic field is: " + str(mag))
    mag = mag.rotate(q.r33)
    print("  Rotated magnetic field is:   " + str(mag))
    g = Vector3D(0,0,1)
    print("  Gravity is:                  " + str(g))
    g = g.rotate(q.r33)
    print("  Rotated Gravity is:          " + str(g))
    g = q2gravity(q)
    print("  Gravity from q is:           " + str(g))
    print("  Gravity is:                  " + str(g))
    print("  Magnetic field is:           " + str(mag))
    print("  Quaternion should be:        " + str(q))
    print("  Quaternion from acc,mag is:  " + str(accelmag2q(g, mag))) #
    print("  RPY should be:               " + str(q2rpy(q)*RAD2DEG))
    print("  RPY should be:               30, 60, -30")                #
    print("  RPY from acc,mag is:        " + str(q2rpy(accelmag2q(g, mag))*RAD2DEG)) # this now include accmag2rpy and rpy2q
    print("  RPY from acc,mag is direct: " + str(accelmag2rpy(g, mag)*RAD2DEG))
    print("  Heading is: " + str(heading(q=q, mag=mag, declination=0.0)*RAD2DEG)) #
    
    print('===========================================================================================')
    print('=================Motion====================================================================')
    print('===========================================================================================')

    
    ######################################
    # Speed Tests
    ######################################

    ######################################
    # f /= norm
    # f = f/norm
    ######################################
    repeat = 100000

    r=random.uniform(0, 5.0)
    f = np.array([r,r,r])
    tic = time.perf_counter()
    for i in range(repeat):
        f /= math.sqrt(f[0]*f[0]+f[1]*f[1]+f[2]*f[2])
    toc = time.perf_counter()
    print(f"f/= value {(toc-tic)/repeat*1000000:0.4f} micro seconds")  

    tic = time.perf_counter()
    for i in range(repeat):
        f = f / math.sqrt(f[0]*f[0]+f[1]*f[1]+f[2]*f[2])
    toc = time.perf_counter()
    print(f"f=f/value {(toc-tic)/repeat*1000000:0.4f} micro seconds")  

    ######################################
    # linalg.norm versus sqrt(f0**2+f1**2+f2**2)
    ######################################
    repeat = 100000

    r=random.uniform(0, 5.0)
    f = np.array([r,r,r])
    tic = time.perf_counter()
    for i in range(repeat):
        k=f/np.linalg.norm(f)
    toc = time.perf_counter()
    print(f"linalg.norm {(toc-tic)/repeat*1000000:0.4f} micro seconds")  

    tic = time.perf_counter()
    for i in range(repeat):
        k=f/math.sqrt(f[0]*f[0]+f[1]*f[1]+f[2]*f[2])
    toc = time.perf_counter()
    print(f"sqrt(f0**f0...) {(toc-tic)/repeat*1000000:0.4f} micro seconds")  

    f = np.array([r,r,r,r,r,r])
    tic = time.perf_counter()
    for i in range(repeat):
        k=f/np.linalg.norm(f)
    toc = time.perf_counter()
    print(f"linalg.norm {(toc-tic)/repeat*1000000:0.4f} micro seconds")  

    tic = time.perf_counter()
    for i in range(repeat):
        k=f/math.sqrt(f[0]*f[0]+f[1]*f[1]+f[2]*f[2]+f[3]*f[3]+f[4]*f[4]+f[5]*f[5])
    toc = time.perf_counter()
    print(f"sqrt(f0**f0...5) {(toc-tic)/repeat*1000000:0.4f} micro seconds")  
 
    ######################################
    # The famous x*x versus x**2
    # It looks like x*x is 10% faster
    ######################################

    repeat = 1000000

    tic = time.perf_counter()
    for i in range(repeat):
        x = random.uniform(0, 5.0)
    toc = time.perf_counter()
    rand_time = toc-tic

    tic = time.perf_counter()
    for i in range(repeat):
        x = random.uniform(0, 5.0)
        tmp = x**2
    toc = time.perf_counter()
    print(f"x**2 {(toc-tic-rand_time)/repeat*1000000:0.4f} micro seconds")  

    tic = time.perf_counter()
    for i in range(repeat):
        x = random.uniform(4.0, 5.0)
        tmp = x*x
    toc = time.perf_counter()
    print(f"x*x {(toc-tic-rand_time)/repeat*1000000:0.4f} micro seconds")  


    ######################################
    # Is it faster to preallocate matrix 
    # and then assign each element or use 
    # directly np.array()
    ######################################
    
    repeat = 100000
    tic = time.perf_counter()
    for i in range(repeat):
        xx = random.uniform(0., 1.)
        xy = random.uniform(0., 1.)
        xz = random.uniform(0., 1.)
        xw = random.uniform(0., 1.)
        yy = random.uniform(0., 1.)
        yz = random.uniform(0., 1.)
        yw = random.uniform(0., 1.)
        zz = random.uniform(0., 1.)
        zw = random.uniform(0., 1.)
    toc = time.perf_counter()
    rand_time =  toc-tic

    tic = time.perf_counter()
    for i in range(repeat):
        xx = random.uniform(0., 1.)
        xy = random.uniform(0., 1.)
        xz = random.uniform(0., 1.)
        xw = random.uniform(0., 1.)
        yy = random.uniform(0., 1.)
        yz = random.uniform(0., 1.)
        yw = random.uniform(0., 1.)
        zz = random.uniform(0., 1.)
        zw = random.uniform(0., 1.)

        # Construct the rotation matrix
        r33 = np.empty((3,3))
        r33[0,0] = 1.0 - (yy + zz)
        r33[0,1] =       (xy - zw)
        r33[0,2] =       (xz + yw)
        r33[1,0] =       (xy + zw)
        r33[1,1] = 1.0 - (xx + zz)
        r33[1,2] =       (yz - xw)
        r33[2,0] =       (xz - yw)
        r33[2,1] =       (yz + xw)
        r33[2,2] = 1.0 - (xx + yy)
    toc = time.perf_counter()
    print(f"r33 prealloc {(toc-tic-rand_time)/repeat*1000000:0.4f} micro seconds")  
       
    tic = time.perf_counter()
    for i in range(repeat):
        xx = random.uniform(0., 1.)
        xy = random.uniform(0., 1.)
        xz = random.uniform(0., 1.)
        xw = random.uniform(0., 1.)
        yy = random.uniform(0., 1.)
        yz = random.uniform(0., 1.)
        yw = random.uniform(0., 1.)
        zz = random.uniform(0., 1.)
        zw = random.uniform(0., 1.)

        # Construct the rotation matrix
        r33 = np.array([
            [1.0 - (yy + zz), (xy - zw), (xz + yw)],
            [(xy + zw), 1.0 - (xx + zz), (yz - xw)],
            [(xz - yw), (yz + xw), 1.0 - (xx + yy)]
        ])
    toc = time.perf_counter()
    print(f"r33 direct {(toc-tic-rand_time)/repeat*1000000:0.4f} micro seconds")  

    # # Make sure this is actually the same
    # xx = random.uniform(0., 1.)
    # xy = random.uniform(0., 1.)
    # xz = random.uniform(0., 1.)
    # xw = random.uniform(0., 1.)
    # yy = random.uniform(0., 1.)
    # yz = random.uniform(0., 1.)
    # yw = random.uniform(0., 1.)
    # zz = random.uniform(0., 1.)
    # zw = random.uniform(0., 1.)

    # # Construct the rotation matrix
    # r33 = np.empty((3,3))
    # r33[0,0] = 1.0 - (yy + zz)
    # r33[0,1] =       (xy - zw)
    # r33[0,2] =       (xz + yw)
    # r33[1,0] =       (xy + zw)
    # r33[1,1] = 1.0 - (xx + zz)
    # r33[1,2] =       (yz - xw)
    # r33[2,0] =       (xz - yw)
    # r33[2,1] =       (yz + xw)
    # r33[2,2] = 1.0 - (xx + yy)

    # r33c = np.array([
    # [1.0 - (yy + zz), (xy - zw), (xz + yw)],
    # [(xy + zw), 1.0 - (xx + zz), (yz - xw)],
    # [(xz - yw), (yz + xw), 1.0 - (xx + yy)]
    # ])

    ######################################
    # Is it faster to multiply quaternion 
    # by sqrt(2) compare to multiply 
    # elements later with 2.
    ######################################
    
    SQRT2 = math.sqrt(2.0)
    tic = time.perf_counter()
    for i in range(repeat):
        q_sqrt2 = q*SQRT2
        xx = q_sqrt2.x**2
        xy = q_sqrt2.x * q_sqrt2.y
        xz = q_sqrt2.x * q_sqrt2.z
        xw = q_sqrt2.x * q_sqrt2.w
        yy = q_sqrt2.y**2
        yz = q_sqrt2.y * q_sqrt2.z
        yw = q_sqrt2.y * q_sqrt2.w
        zz = q_sqrt2.z**2
        zw = q_sqrt2.z * q_sqrt2.w
        r33= np.array([
            [1.0 - (yy + zz),          (xy - zw),          (xz + yw)],
            [      (xy + zw),    1.0 - (xx + zz),          (yz - xw)],
            [      (xz - yw),          (yz + xw),    1.0 - (xx + yy)]
        ]) 
    toc = time.perf_counter()
    print(f"q 2 r33 sqrt2 speedup {(toc-tic)/repeat*1000000:0.4f} micro seconds")  
    
    tic = time.perf_counter()
    for i in range(repeat):
        xx = q.x**2
        xy = q.x * q.y
        xz = q.x * q.z
        xw = q.x * q.w
        yy = q.y**2
        yz = q.y * q.z
        yw = q.y * q.w
        zz = q.z**2
        zw = q.z * q.w
        r33= np.array([
            [1.0 - 2.*(yy + zz),          2.*(xy - zw),          2.*(xz + yw)],
            [      2.*(xy + zw),    1.0 - 2.*(xx + zz),          2.*(yz - xw)],
            [      2.*(xz - yw),          2.*(yz + xw),    1.0 - 2.*(xx + yy)]
        ]) 
    toc = time.perf_counter()
    print(f"q 2 r33 no sqrt2 speedup {(toc-tic)/repeat*1000000:0.4f} micro seconds")  
    
    # # make sure this is the same
    # SQRT2 = math.sqrt(2.0)
    # q_sqrt2 = q*SQRT2
    # xx = q_sqrt2.x**2
    # xy = q_sqrt2.x * q_sqrt2.y
    # xz = q_sqrt2.x * q_sqrt2.z
    # xw = q_sqrt2.x * q_sqrt2.w
    # yy = q_sqrt2.y**2
    # yz = q_sqrt2.y * q_sqrt2.z
    # yw = q_sqrt2.y * q_sqrt2.w
    # zz = q_sqrt2.z**2
    # zw = q_sqrt2.z * q_sqrt2.w
    # r33= np.array([
    #     [1.0 - (yy + zz),          (xy - zw),          (xz + yw)],
    #     [      (xy + zw),    1.0 - (xx + zz),          (yz - xw)],
    #     [      (xz - yw),          (yz + xw),    1.0 - (xx + yy)]
    # ]) 
    # print(r33)
    
    # xx = q.x**2
    # xy = q.x * q.y
    # xz = q.x * q.z
    # xw = q.x * q.w
    # yy = q.y**2
    # yz = q.y * q.z
    # yw = q.y * q.w
    # zz = q.z**2
    # zw = q.z * q.w
    # r33c= np.array([
    #     [1.0 - 2.*(yy + zz),          2.*(xy - zw),          2.*(xz + yw)],
    #     [      2.*(xy + zw),    1.0 - 2.*(xx + zz),          2.*(yz - xw)],
    #     [      2.*(xz - yw),          2.*(yz + xw),    1.0 - 2.*(xx + yy)]
    # ]) 
    # print(r33c)
    

    ######################################
    # Is rotate with r33 faster than
    # q.conjugate * Quaternion(g) * q
    ######################################
    
    repeat = 100000
    q = rpy2q(30*DEG2RAD,60*DEG2RAD,-30*DEG2RAD) # make plane point straight forward
    g = Vector3D(0,0,1)
    m = Vector3D(1,0,1)                      # make magnetic field point forward and down

    tic = time.perf_counter()
    for i in range(repeat):
        h = m.rotate(q.r33.T)                                       # (eq. 45)
    toc = time.perf_counter()
    print(f"m.rotate(q.r33.T) took {(toc-tic)/repeat*1000000:0.4f} micro seconds")
    
    tic = time.perf_counter()
    for i in range(repeat):
        h = q * m * q.conjugate                                  # (eq. 45)
    toc = time.perf_counter()
    print(f"q * m * q.conjugate took {(toc-tic)/repeat*1000000:0.4f} micro seconds")

    tic = time.perf_counter()
    for i in range(repeat):
        q2rpy(rpy2q(r=90*DEG2RAD,p=0.0,y=0.0))
    toc = time.perf_counter()
    print(f"q2rpy(rpy2q(r=90*DEG2RAD,p=0.0,y=0.0)) took {(toc-tic)/repeat*1000000:0.4f} micro seconds")

    tic = time.perf_counter()
    for i in range(repeat):
        q=rpy2q(0,90*DEG2RAD,0)
    toc = time.perf_counter()
    print(f"rpy2q(0,90*DEG2RAD,0) took {(toc-tic)/repeat*1000000:0.4f} micro seconds")

    ######################################
    # Is q to gravity faster than
    # q.conjugate * Quaternion(g) * q
    # or g.ratete(q.r33)
    # or q.r33.T@g
    ######################################

    tic = time.perf_counter()
    for i in range(repeat):
        q2gravity(q)
    toc = time.perf_counter()
    print(f"q2gravity(q) took {(toc-tic)/repeat*1000000:0.4f} micro seconds")

    tic = time.perf_counter()
    for i in range(repeat):
        (q.conjugate * Quaternion(g) * q).v
    toc = time.perf_counter()
    print(f"(q.conjugate * Quaternion(g) * q).v took {(toc-tic)/repeat*1000000:0.4f} micro seconds")

    tic = time.perf_counter()
    for i in range(repeat):
        g.rotate(q.r33)
    toc = time.perf_counter()
    print(f"g.rotate(q.r33) took {(toc-tic)/repeat*1000000:0.4f} micro seconds")  

    tic = time.perf_counter()
    g=g.v
    for i in range(repeat):
        (q.r33).T@g
    toc = time.perf_counter()
    print(f"q.r33.T@g took {(toc-tic)/repeat*1000000:0.4f} micro seconds")  


    ######################################
    # The famous fast inverse square root
    # https://en.wikipedia.org/wiki/Fast_inverse_square_root
    # x**-0.5 is the fastest on x64/86 architecture!
    ######################################

    repeat = 1000000

    tic = time.perf_counter()
    for i in range(repeat):
        x = random.uniform(4.0, 5.0)
    toc = time.perf_counter()
    rand_time = toc-tic

    tic = time.perf_counter()
    for i in range(repeat):
        x = random.uniform(4.0, 5.0)
        tmp = x**-0.5
    toc = time.perf_counter()
    print(f"x**-0.5 {(toc-tic-rand_time)/repeat*1000000:0.4f} micro seconds")  

    from pyIMU.utilities import invSqrt
    tic = time.perf_counter()
    for i in range(repeat):
        x = random.uniform(4.0, 5.0)
        tmp = invSqrt(x)
    toc = time.perf_counter()
    print(f"pyIMU.utilities.invSqrt {(toc-tic-rand_time)/repeat*1000000:0.4f} micro seconds")  

    try:
        import mpmath
        old_repeat = repeat
        repeat = 100000
        mpmath.mp.dps = 4
        tic = time.perf_counter()
        for i in range(repeat):
            x = random.uniform(4.0, 5.0)
            tmp = 1./mpmath.sqrt(x)
        toc = time.perf_counter()
        print(f"mpmath.sqrt {(toc-tic-rand_time)/repeat*1000000:0.4f} micro seconds")  
        repeat = old_repeat
    except:
        print("mpmath not installed")
        
    try:
        from fastmath import invSqrt, invSqrt_fast

        tic = time.perf_counter()
        for i in range(repeat):
            x = random.uniform(4.0, 5.0)
            tmp = invSqrt(x)
        toc = time.perf_counter()
        print(f"fastmath.invSqrt {(toc-tic-rand_time)/repeat*1000000:0.4f} micro seconds")  

        tic = time.perf_counter()
        for i in range(repeat):
            x = random.uniform(4.0, 5.0)
            tmp = invSqrt_fast(x)
        toc = time.perf_counter()
        print(f"fastmath.invSqrt_fast {(toc-tic-rand_time)/repeat*1000000:0.4f} micro seconds")  

    except:
        print("fastmath not installed")


