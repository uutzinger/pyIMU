import time
from pyIMU.quaternion import Quaternion, Vector3D, DEG2RAD, EPSILON 
from pyIMU.utilities  import invSqrt, accel2rpy, accel2q, q2rpy, rpy2q, q2gravity
from scipy.spatial.transform import Rotation
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
    # Converting rpy to quaternion and then back to rpy:
    print("Roll 90, Pitch 0, Yaw 0")
    print("Following should be 1.5708, 0.0, 0.0")
    print(str(q2rpy(rpy2q(r=90*DEG2RAD,p=0.0,y=0.0))))
    print("Roll 0, Pitch 90, Yaw 0")
    print("Following should be 0.0, 1.5708, 0.0")
    print("But its pi, 1.5708, pi which is the same")
    print(str(q2rpy(rpy2q(r=0.0,p=90*DEG2RAD,y=0.0))))    # sort of same 
    print("Roll 0, Pitch 0, Yaw 90")    
    print("Following should be 0.0, 0.0, 1.5708")
    print(str(q2rpy(rpy2q(r=0.0,p=0.0,y=90*DEG2RAD))))    # pass
    
    gravity = Vector3D(0,0,1)
    print("Gravity vector in North East Down:")
    print(str(gravity))

    # Roll 90 degrees, points y axis downward, 
    # gravity (0,0,1) should now be on y axis 0,1,0
    print("Rotate system 90 degrees clockwise around X (North)")
    q = rpy2q(r=90*DEG2RAD,p=0.0,y=0.0)
    print("The following 3 lines should be: 0,1,0")
    print("q.T * Gravity * q: " + str((q.conjugate * Quaternion(gravity) * q).v)) # ok
    print("q2gravity        : " + str(q2gravity(q)))                              # ok
    print("R33 * gravity    : " + str(gravity.rotate(q.r33)))                     # ok

    # Roll -90 degrees, points y axis downward, 
    # gravity (0,0,1) should now be on y axis 0,-1,0
    print("Rotate system 90 degrees counter clockwise around X (North)")
    q = rpy2q(r=-90*DEG2RAD,p=0.0,y=0.0)
    print("The following 3 lines should be: 0,-1,0")
    print("q.T * Gravity * q" + str((q.conjugate * Quaternion(gravity) * q).v)) # ok
    print("q2gravity        " + str(q2gravity(q)))                              # ok
    print("R33 * gravity    " + str(gravity.rotate(q.r33)))                     # ok

    # Pitch 90 degrees, points x upwards
    # gravity (0,0,1) should now be on x axis backwards -1,0,0
    
    print("Rotate system 90 degrees clockwise around Y (East), Pitch, points nose up")
    print("The following 3 lines should be: -1,0,0")
    q = rpy2q(r=0.0,p=90*DEG2RAD,y=0.0)
    print("q.T * Gravity * q" + str((q.conjugate * Quaternion(gravity) * q).v)) # ok
    print("q2gravity        " + str(q2gravity(q)))                              # ok
    print("R33 * Gravity    " + str(gravity.rotate(q.r33)))                     # ok

    # Pitch 90 degrees, points x upwards
    # gravity (0,0,1) should now be on x axis backwards -1,0,0
    print("Rotate system 90 degrees counter clockwise around Y (East), Pitch, points nose up")
    print("The following 3 lines should be: 1,0,0")
    q = rpy2q(r=0.0,p=-90*DEG2RAD,y=0.0)
    print("q.T * Gravity * q" + str((q.conjugate * Quaternion(gravity) * q).v)) # ok
    print("q2gravity        " + str(q2gravity(q)))                              # ok
    print("R33 * Gravity    " + str(gravity.rotate(q.r33)))                     # ok

    # Yaw 90 degrees, x becomes y and y becomes -x,
    # gravity (0,0,1) should remain same 0,0,1
    print("Rotate system 90 degrees clockwise around Z (Down), Yaw, nose to right")
    print("The following 3 lines should be: 0,0,1")
    q = rpy2q(r=0.0,p=0.0,y=90*DEG2RAD)
    print("q.T * Gravity * q" + str((q.conjugate * Quaternion(gravity) * q).v)) # ok
    print("q2gravity        " + str(q2gravity(q)))                              # ok
    print("R33 * Gravity    " + str(gravity.rotate(q.r33)))                     # ok

    # Yaw 90 degrees, x becomes y and y becomes -x,
    # gravity (0,0,1) should remain same 0,0,1
    print("Rotate system 90 degrees clockwise around Z (Down), Yaw, nose to left")
    print("The following 3 lines should be: 0,0,1")
    q = rpy2q(r=0.0,p=0.0,y=-90*DEG2RAD)
    print("q.T * Gravity * q" + str((q.conjugate * Quaternion(gravity) * q).v)) # ok
    print("q2gravity        " + str(q2gravity(q)))                              # ok
    print("R33 * Gravity    " + str(gravity.rotate(q.r33)))                     # ok

    # No rotation
    # gravity (0,0,1) should remain same 0,0,1
    print("Rotate system 0 degrees")
    print("The following 3 lines should be: 0,0,1")
    q = rpy2q(r=0.0,p=0.0,y=0.0)
    print("q.T * Gravity * q" + str((q.conjugate * Quaternion(gravity) * q).v)) # ok
    print("q2gravity        " + str(q2gravity(q)))                       # ok
    print("R33 * Gravity    " + str(gravity.rotate(q.r33)))              # ok
    
    # Converting gravity to rpy should result in 0,0,0 
    # rpy of 0,0,0 converted to quaternion should result in ?
    # and quaternion converted back to rpy should result in 0,0,0
    print("Convert gravity to RPY and back to quaternion and back to RPY")
    print("The following line should be: 0,0,0")
    r=accel2rpy(gravity) # Should result in r=0,p=0,(y=0): ok
    print(str(r))
    print("The following 2 lines should be: 1,0,0,0")
    q = rpy2q(r=0.0,p=0.0,y=0.0)
    print(str(q))
    print(str(accel2q(gravity))) # should be the same as previous print : ok
    print("The following line should be: 0,0,0")
    print(str(q2rpy(q))) # Should result r=0,p=0,(y=0): ok
    # Gravity to quaternion should be same as rpy2q of 0,0,0
    print("The following 2 lines should be: 1,0,0,0")
    print(str(accel2q(gravity)))                # ok
    print(str(rpy2q(r=0.0,p=0.0,y=0.0)))        # ok

    # Compare this code with online calculators
    # Google online quaternion converter
    # Rotation Matrix is cosine matrix
    # Roll(Bank) Pitch Yaw    
    
    # RPY (X,Y,Z)          0,0,0                    Wolfram Alpha RPY (1-2-3)
    # Quaternion (w,x,y,z) 1,0,0,0                  quaternions.online, energid, ninja, topo, matlab, this code
    # Quaternion (w,x,y,z) 1,NA,NA,NA               Wolfram Alpha
    # Rotation Matrix      [1,0,0],[0,1,0],[0,0,1]  geschler, ninja, matlab, this code
    # Angle Axis (a,x,y,z)  0,1,0,0                 ninja
    # Angle Axis            0,0,0,0                 energid, geschler  
    print("Compare")
    q=rpy2q(0,0,0)
    print("The following lines should be: 1,0,0,0")
    print(str(q))
    print("The following lines should be: [1,0,0],[0,1,0],[0,0,1]")
    print(str(q.r33))

    roll, pitch, yaw = 0, 0, 0
    r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    quaternion = r.as_quat()
    print("Scipy quaternion is:")
    print(quaternion)
    print("Scipy quaternion is not explained in documentation, so likely representation is x,y,z,w?")

    # RPY (X,Y,Z)          90,0,0                   Wolfram Alpha RPY (1-2-3)
    # Quaternion (w,x,y,z) 0.7,0.7,0,0              Wolfram Alpha, quaternions.online, gaschler, energid, ninja, matlab, this code
    # Rotation Matrix      [1,0,0],[0,0,1],[0,-1,0] Wolfram Alpha, ninja, matlab
    # Rotation Matrix      [1,0,0],[0,0,-1],[0,1,0] gaschler, energid, ninja, this code
    # Angle Axis (a,x,y,z) 90,1,0,0                 gaschler, energid, ninja
    q=rpy2q(90*DEG2RAD,0,0)
    print("The following lines should be: 0.7,0.7,0,0")
    print(str(q))
    print("The following lines should be: [1,0,0],[0,0,-1],[0,1,0]")
    print(str(q.r33))
        
    # RPY (X,Y,Z)          0,90,0
    # Quaternion (w,x,y,z) 0.7,0,0.7,0              Wolfram alpha, quaternions.online, gaschler, energid, ninja, topo, matlab, this code
    # Rotation Matrix      [0,0,1],[0,1,0],[-1,0,0] gaschler, energid, ninja, matlab, this code
    # Rotation Matrix      [0,0,-1],[0,1,0],[1,0,0] Wolfram Alpha
    # Angle Axis           90,0,1,0                 energid, ninja
    q=rpy2q(0,90*DEG2RAD,0)
    print("The following lines should be: 0.7, 0, 0.7, 0")
    print(str(q))
    print("The following lines should be: [0, 0, 1],[0, 1, 0],[-1, 0, 0]")
    print(str(q.r33))
    
    # RPY (X,Y,Z)          0,0,90
    # Quaternion (w,x,y,z) 0.7,0,0,0.7              Wolfram alpha, quaternions.online, gaschler, ninja, matlab, this code
    # Rotation Matrix      [0,-1,0],[1,0,0],[0,0,1] gaschler, energid, ninja, this code
    # Rotation Matrix      [0,1,0],[-1,0,0],[0,0,1] Wolfram Alpha
    # Angle Axis           90,0,0,1                 gaschler,energid, ninja
    q=rpy2q(0,0,90*DEG2RAD)
    print("The following lines should be: 0.7, 0, 0, 0.7")
    print(str(q))
    print("The following lines should be: [0,-1,0],[1,0,0],[0,0,1]")
    print(str(q.r33))

    ######################################
    # Speed Tests
    ######################################
    
    gravity = Vector3D(0,0,1)

    repeat = 100000

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

    tic = time.perf_counter()
    for i in range(repeat):
        q2gravity(q)
    toc = time.perf_counter()
    print(f"q2gravity(q) took {(toc-tic)/repeat*1000000:0.4f} micro seconds")

    tic = time.perf_counter()
    for i in range(repeat):
        (q.conjugate * Quaternion(gravity) * q).v
    toc = time.perf_counter()
    print(f"(q.conjugate * Quaternion(gravity) * q).v took {(toc-tic)/repeat*1000000:0.4f} micro seconds")

    tic = time.perf_counter()
    for i in range(repeat):
        gravity.rotate(q.r33)
    toc = time.perf_counter()
    print(f"gravity.rotate(q.r33) took {(toc-tic)/repeat*1000000:0.4f} micro seconds")  

    tic = time.perf_counter()
    g=gravity.v
    for i in range(repeat):
        (q.r33).T@g
    toc = time.perf_counter()
    print(f"q.r33.T@g took {(toc-tic)/repeat*1000000:0.4f} micro seconds")  

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
