import numpy as np

def ellipsoid_fit(X, flag=0, equals=None):
    '''
    Fit an ellispoid/sphere to a set of xyz data points:

      [center, radii, evecs, pars ] = ellipsoid_fit( X )
      [center, radii, evecs, pars ] = ellipsoid_fit( [x y z] );
      [center, radii, evecs, pars ] = ellipsoid_fit( X, 1 );
      [center, radii, evecs, pars ] = ellipsoid_fit( X, 2, 'xz' );
      [center, radii, evecs, pars ] = ellipsoid_fit( X, 3 );

    Parameters:
    * X=[x y z]    - Cartesian data, n x 3 matrix or three n x 1 vectors
    * flag         - 0 fits an arbitrary ellipsoid (default),
                   - 1 fits an ellipsoid with its axes along [x y z] axes
                   - 2 followed by, say, 'xy' fits as 1 but also x_rad = y_rad
                   - 3 fits a sphere

    Output:
    * center    -  ellipsoid center coordinates [xc; yc; zc]
    * ax        -  ellipsoid radii [a; b; c]
    * evecs     -  ellipsoid radii directions as columns of the 3x3 matrix
    * v         -  the 9 parameters describing the ellipsoid algebraically: 
                   Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz = 1
                   
    Author:
    Yury Petrov, Northeastern University, Boston, MA
    '''
    
    # Check input arguments
    if equals is None:
        equals = 'xy'
    
    if X.shape[1] != 3:
        raise ValueError('Input data must have three columns!')
    else:
        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]
    
    # Need nine or more data points
    if len(x) < 9 and flag == 0:
        raise ValueError('Must have at least 9 points to fit a unique ellipsoid')
    if len(x) < 6 and flag == 1:
        raise ValueError('Must have at least 6 points to fit a unique oriented ellipsoid')
    if len(x) < 5 and flag == 2:
        raise ValueError('Must have at least 5 points to fit a unique oriented ellipsoid with two axes equal')
    if len(x) < 3 and flag == 3:
        raise ValueError('Must have at least 4 points to fit a unique sphere')
    
    if flag == 0:
        # Fit ellipsoid in the form Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz = 1
        D = np.column_stack((x * x, y * y, z * z, 2 * x * y, 2 * x * z, 2 * y * z, 2 * x, 2 * y, 2 * z))
    elif flag == 1:
        # Fit ellipsoid in the form Ax^2 + By^2 + Cz^2 + 2Gx + 2Hy + 2Iz = 1
        D = np.column_stack((x * x, y * y, z * z, 2 * x, 2 * y, 2 * z))
    elif flag == 2:
        # Fit ellipsoid in the form Ax^2 + By^2 + Cz^2 + 2Gx + 2Hy + 2Iz = 1, where A = B or B = C or A = C
        if equals == 'yz' or equals == 'zy':
            D = np.column_stack((y * y + z * z, x * x, 2 * x, 2 * y, 2 * z))
        elif equals == 'xz' or equals == 'zx':
            D = np.column_stack((x * x + z * z, y * y, 2 * x, 2 * y, 2 * z))
        else:
            D = np.column_stack((x * x + y * y, z * z, 2 * x, 2 * y, 2 * z))
    else:
        # Fit sphere in the form A(x^2 + y^2 + z^2) + 2Gx + 2Hy + 2Iz = 1
        D = np.column_stack((x * x + y * y + z * z, 2 * x, 2 * y, 2 * z))
    
    # Solve the normal system of equations
    v = np.linalg.lstsq(D.T @ D, D.T @ np.ones(len(x)), rcond=None)[0]
    
    # Find the ellipsoid parameters
    if flag == 0:
        # Form the algebraic form of the ellipsoid
        A = np.array([[v[0], v[3], v[4], v[6]],
                      [v[3], v[1], v[5], v[7]],
                      [v[4], v[5], v[2], v[8]],
                      [v[6], v[7], v[8], -1]])
        
        # Find the center of the ellipsoid
        center = -np.linalg.inv(A[:3, :3]) @ np.array([v[6], v[7], v[8]])
        
        # Form the corresponding translation matrix
        T = np.eye(4)
        T[:3, 3] = center
        
        # Translate to the center
        R = T @ A @ T.T
        
        # Solve the eigenproblem
        evals, evecs = np.linalg.eig(-R[:3, :3] / R[3, 3])
        radii = np.sqrt(1 / np.diag(evals))
    else:
        if flag == 1:
            v = np.array([v[0], v[1], v[2], 0, 0, 0, v[3], v[4], v[5]])
        elif flag == 2:
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