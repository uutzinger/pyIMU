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



def calibrate(x, y, z):
  H = np.array([x, y, z, -y**2, -z**2, np.ones([len(x), 1])])
  H = np.transpose(H)
  w = x**2
  
  (X, residues, rank, shape) = linalg.lstsq(H, w)
  
  OSx = X[0] / 2
  OSy = X[1] / (2 * X[3])
  OSz = X[2] / (2 * X[4])
  
  A = X[5] + OSx**2 + X[3] * OSy**2 + X[4] * OSz**2
  B = A / X[3]
  C = A / X[4]
  
  SCx = np.sqrt(A)
  SCy = np.sqrt(B)
  SCz = np.sqrt(C)
  
  # type conversion from numpy.float64 to standard python floats
  offsets = [OSx, OSy, OSz]
  scale = [SCx, SCy, SCz]
  
  offsets = map(np.asscalar, offsets)
  scale = map(np.asscalar, scale)
  
  return (offsets, scale)


def calibrate_from_file(file_name):
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

  return calibrate(np.array(samples_x), np.array(samples_y), np.array(samples_z))


def compute_calibrate_data(data, offsets, scale):
  output = [[], [], []]
  for i in range(len(data[0])):
    output[0].append((data[0][i] - offsets[0]) / scale[0])
    output[1].append((data[1][i] - offsets[1]) / scale[1])
    output[2].append((data[2][i] - offsets[2]) / scale[2])
  return output


def ellipsoid_fit(X, **kwargs):
    '''
    Fit an ellispoid/sphere to a set of xyz data points:

      [center, radii, evecs, pars ] = ellipsoid_fit( X )
      [center, radii, evecs, pars ] = ellipsoid_fit( np.array([x y z]) );
      [center, radii, evecs, pars ] = ellipsoid_fit( X, 1 );
      [center, radii, evecs, pars ] = ellipsoid_fit( X, 2, 'xz' );
      [center, radii, evecs, pars ] = ellipsoid_fit( X, 3 );

    Parameters:
    * X=[x y z]    - Cartesian data, n x 3 matrix or three n x 1 vectors
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
    
    if X.shape[1] != 3:
        raise ValueError('Input data must have three columns!')
    else:
        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]
    
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
  
  print "Calibrating from acc.txt"
  (offsets, scale) = calibrate_from_file("acc.txt")
  print "Offsets:"
  print offsets
  print "Scales:"
  print scale
  
  print "Calibrating from magn.txt"
  (offsets, scale) = calibrate_from_file("magn.txt")
  print "Offsets:"
  print offsets
  print "Scales:"
  print scale