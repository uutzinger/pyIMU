"""
cal_lib.py - Ellipsoid into Sphere calibration library based upon numpy and linalg
Copyright (C) 2012 Fabio Varesano <fabio at varesano dot net>

Updates by Urs Utzinger to include ellipsoid fit from Yury Petrov <yurypetrov at gmail dot com>

Development of this code has been supported by the Department of Computer Science,
Universita' degli Studi di Torino, Italy within the Piemonte Project
http://www.piemonte.di.unito.it/

This program is free software: you can redistribute it and/or modify
it under the terms of the version 3 GNU General Public License as
published by the Free Software Foundation.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from numpy import linalg

def calibrate(data):
    
    (center, radii, evecs, v)= ellipsoid_fit(data)
 
    scaleMat = linalg.inv(np.array([[radii(1), 0, 0], [0, radii(2), 0], [0, 0, radii(3)]]) * min(radii)) 
    
    correctionMat = evecs * scaleMat * evecs.T
        
    # now correct the data to show that it works

    magVector = np.array([x - center(1), y - center(2), z - center(3)]).T # take off center offset
    # magVector = correctionMat * magVector;				              # do rotation and scale
    magVector = evecs * magVector				                          # do rotation and scale
    xCorr = magVector[0, :]					                              # get corrected vectors
    yCorr = magVector[1, :]
    zCorr = magVector[2, :]

    print('{:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f}'.format(
        center(1), center(2), center(3),
        correctionMat(1, 1), correctionMat(1, 2), correctionMat(1, 3),
        correctionMat(2, 1), correctionMat(2, 2), correctionMat(2, 3),
        correctionMat(3, 1), correctionMat(3, 2), correctionMat(3, 3)))

    # scale = np.diag(correctionMat)
    
    return (center, correctionMat)

    # H = np.array([x, y, z, -y**2, -z**2, np.ones([len(x), 1])])
    # H = np.transpose(H)
    # w = x**2

    # # solving H * a = w for a
    # # a0*x + a1*y + a2*z - a3*y^2 - a4*z^2 -x^2 = -a5
    # # Ax^2 + By^2 + Cz^2 + 2Gx + 2Hy + 2Iz = 1

    # (X, residues, rank, shape) = linalg.lstsq(H, w)

    # OSx = X[0] / 2
    # OSy = X[1] / (2 * X[3])
    # OSz = X[2] / (2 * X[4])

    # A = X[5] + OSx**2 + X[3] * OSy**2 + X[4] * OSz**2
    # B = A / X[3]
    # C = A / X[4]

    # SCx = np.sqrt(A)
    # SCy = np.sqrt(B)
    # SCz = np.sqrt(C)

    # # type conversion from numpy.float64 to standard python floats
    # offsets = [OSx, OSy, OSz]
    # scale   = [SCx, SCy, SCz]

    # offsets = map(np.asscalar, offsets)
    # scale   = map(np.asscalar, scale)

    # return (offsets, scale)

def calibrate_from_file(file_name):
    '''read data from file, each line has 3 floats'''
    samples_f = open(file_name, 'r')
    samples_x = []
    samples_y = []
    samples_z = []
    for line in samples_f:
        reading = line.split()
        if len(reading) == 3:
            samples_x.append(int(reading[0]))
            samples_y.append(int(reading[1]))
            samples_z.append(int(reading[2]))
    
    (center, correctionMat) = calibrate(np.array([samples_x, samples_y, samples_z]))
    
    return (center, correctionMat)

def compute_calibrate_data(data, offsets, correctionMat):
    
    output = None

    if isinstance(data, np.ndarray):
        if data.shape[-1] == 3:
            if isinstance(offsets, np.ndarray):
                o_shape = offsets.shape
                if offsets.shape[-1] == 3 and len(o_shape)==1:
                    if data is not None:
                        output = data - offsets # subtract offsets

    if isinstance(correctionMat, np.ndarray):
        c_shape = correctionMat.shape
        if len(c_shape) == 2:
            if c_shape[0] == 3 and c_shape[1] == 3:
                if output is not None:
                    output = output @ correctionMat # apply scale and cross axis sensitivity

    return output


def ellipsoid_fit(data, **kwargs):
    '''
    Fit an ellipsoid/sphere to a set of xyz data points:

      [center, radii, evecs, pars ] = ellipsoid_fit( data )
      [center, radii, evecs, pars ] = ellipsoid_fit( np.array([x y z]) );
      [center, radii, evecs, pars ] = ellipsoid_fit( data, 1 );
      [center, radii, evecs, pars ] = ellipsoid_fit( data, 2, 'xz' );
      [center, radii, evecs, pars ] = ellipsoid_fit( data, 3 );

    Parameters:
    * data=[x y z]    - Cartesian data, n x 3 matrix or three n x 1 vectors
    * option       - 0 fits an arbitrary ellipsoid (default),
                   - 1 fits an ellipsoid with its axes along [x y z] axes
                   - 2 followed by, say, 'xy' fits as 1 but also x_rad = y_rad
                   - 3 fits a sphere
    * equals       - 'xy', 'xz', or 'yz' for option==2

    Output:
    * center    -  ellipsoid center coordinates [xc; yc; zc]
    * ax        -  ellipsoid radii [a; b; c]
    * evecs     -  ellipsoid radii directions as columns of the 3x3 matrix
    * v         -  the 9 parameters describing the ellipsoid algebraically: 
                   Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz = 1

    This needs: numpy
    
    Author:
    Yury Petrov, Northeastern University, Boston, MA
    '''

    option  = kwargs.get('option', 0)
    equals  = kwargs.get('equals', 'xy')
    
    if data.shape[1] != 3:
        raise ValueError('Input data must have three columns!')
    else:
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
    
    # Need nine or more data points
    if len(x) < 9 and option == 0:
        raise ValueError('Must have at least 9 points to fit a unique ellipsoid')
    if len(x) < 6 and option == 1:
        raise ValueError('Must have at least 6 points to fit a unique oriented ellipsoid')
    if len(x) < 5 and option == 2:
        raise ValueError('Must have at least 5 points to fit a unique oriented ellipsoid with two axes equal')
    if len(x) < 3 and option == 3:
        raise ValueError('Must have at least 4 points to fit a unique sphere')
    
    if   option == 0: # Fit ellipsoid in the form Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz = 1
        D = np.column_stack((x*x, y*y, z*z, 2*x*y, 2*x*z, 2*y*z, 2*x, 2*y, 2*z))
    elif option == 1: # Fit ellipsoid in the form Ax^2 + By^2 + Cz^2 + 2Gx + 2Hy + 2Iz = 1
        D = np.column_stack((x*x, y*y, z*z, 2*x, 2*y, 2*z))
    elif option == 2: # Fit ellipsoid in the form Ax^2 + By^2 + Cz^2 + 2Gx + 2Hy + 2Iz = 1, where A = B or B = C or A = C
        if equals == 'yz' or equals == 'zy':
            D = np.column_stack((y*y+z*z, x*x, 2*x, 2*y, 2*z))
        elif equals == 'xz' or equals == 'zx':
            D = np.column_stack((x*x+z*z, y*y, 2*x, 2*y, 2*z))
        else:
            D = np.column_stack((x*x+y*y, z*z, 2*x, 2*y, 2*z))
    else:
        # Fit sphere in the form A(x^2 + y^2 + z^2) + 2Gx + 2Hy + 2Iz = 1
        D = np.column_stack((x*x+y*y+z*z, 2*x, 2*y, 2*z))
    
    # Solve the normal system of equations
    v = np.linalg.lstsq(D.T @ D, D.T @ np.ones(len(x)), rcond=None)[0]
    
    # Find the ellipsoid parameters
    if option == 0:
        # Form the algebraic form of the ellipsoid
        A = np.array([[v[0], v[3], v[4], v[6]],
                      [v[3], v[1], v[5], v[7]],
                      [v[4], v[5], v[2], v[8]],
                      [v[6], v[7], v[8],  -1]])
        
        # Find the center of the ellipsoid
        center = -np.linalg.inv(A[:3, :3]) @ np.array([v[6], v[7], v[8]])
        
        # Form the corresponding translation matrix
        T = np.eye(4)
        T[:3, 3] = center
        
        # Translate to the center
        R = T @ A @ T.T
        
        # Solve the Eigen problem
        evals, evecs = np.linalg.eig(-R[:3, :3] / R[3, 3])
        radii = np.sqrt(1. / np.diag(evals))
    else:
        if option == 1:
            v = np.array([v[0], v[1], v[2], 0, 0, 0, v[3], v[4], v[5]])
        elif option == 2:
            if equals == 'xz' or equals == 'zx':
                v = np.array([v[0], v[1], v[0], 0, 0, 0, v[2], v[3], v[4]])
            elif equals == 'yz' or equals == 'zy':
                v = np.array([v[1], v[0], v[0], 0, 0, 0, v[2], v[3], v[4]])
            else:  # xy
                v = np.array([v[0], v[0], v[1], 0, 0, 0, v[2], v[3], v[4]])
        else:
            v = np.array([v[0], v[0], v[0], 0, 0, 0, v[1], v[2], v[3]])
        
        center = -v[6:9] / v[0:3]
        gam = 1 + (v[6] ** 2 / v[0] + v[7] ** 2 / v[1] + v[8] ** 2 / v[2])
        radii = np.sqrt(gam / v[0:3])
        evecs = np.eye(3)
    
    return center, radii, evecs, v


if __name__ == "__main__":
  
  print("Calibrating from acc.txt")
  (offsets, correctionMat) = calibrate_from_file("acc.txt")
  print("Offsets:")
  print(offsets)
  print("Correction Matrix:")
  print(correctionMat)

  print("Calibrating from gyr.txt")
  (offsets, correctionMat) = calibrate_from_file("gyr.txt")
  print("Offsets:")
  print(offsets)
  print("Correction Matrix:")
  print(correctionMat)
  
  print("Calibrating from mag.txt")
  (offsets, correctionMat) = calibrate_from_file("mag.txt")
  print("Offsets:")
  print(offsets)
  print("Correction Matrix:")
  print(correctionMat)