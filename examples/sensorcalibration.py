'''
    Record Data:
        1. Record Static Values of Gyroscope to obtain bias
        2. Rotate the sensor around each axis (x,y,z) for acceleromter and magnetometer
        3. Rotate the sensor on turntable with several orientations for gyroscope to obtain scale factors: each sensor axis should oscillate at turntable speed

    Fit Data:
        1. Compute gyroscope bias based on static data
        2. Fit the rotation data of the accelerometer to an ellipsoid 
        3. Fit the rotation data of the magnetometer to an ellipsoid       
        4. Obtain scaling factors for gyroscope axes at positive and negative rotations

    Ellipsoid fit
        center, radii, evecs, v= ellipsoid_fit(x,y,z)
        normalized scaling matrix:
            scaleMat = inv([radii(1) 0 0; 0 radii(2) 0; 0 0 radii(3)]) * min(radii)
        cross sensitivity matrix:
            m = evecs * scaleMat * evecs.T

    Save Data:
    	center(1), center(2), center(3)
        [ [m(1, 1), m(1, 2), m(1, 3)],
          [m(2, 1), m(2, 2), m(2, 3)],
          [m(3, 1), m(3, 2), m(3, 3)]]
        
    Correction:
        A. Bias or Center for Accelerometer Magnetometer and Gyroscope
            x=x-center(1)
            y=y-center(2)
            z=z-center(3)
        B. Scaling factor from turntable for Gyroscope
            max is positive rotation
            min is negative rotation
            if x>0 x=x/x.gyr_pos else: x=x/x.gyr_neg
            if y>0 y=y/y.gyr_pos else: y=y/y.gyr_neg
            if z>0 z=z/z.gyr_pos else: z=z/z.gyr_neg
        C. Scaling factors and Cross sensitivity between axes for Accelerometer and Magnetometer
            x=x*m[0,0]+y*m[0,1]+z*m[0,2]
            y=x*m[1,0]+y*m[1,1]+z*m[1,2]
            z=x*m[2,0]+y*m[2,1]+z*m[2,2]

    Accelerometer:    
        Remove Bias
            accVector = [x - center(1), y - center(2), z - center(3)]'
        Rotation and Scale
            accVector = m * accVector

    Magnetometer:    
        Remove Bias
            magVector = [x - center(1), y - center(2), z - center(3)]'
        Rotation and Scale
            magVector = m * magVector
            
    Gyroscope:    
        Remove Bias
            gyrVector = [x - center(1), y - center(2), z - center(3)]'
        Rotation and Scale
            if x>0 x=x/x.gyr_pos else: x=x/x.gyr_neg
            if y>0 y=y/y.gyr_pos else: y=y/y.gyr_neg
            if z>0 z=z/z.gyr_pos else: z=z/z.gyr_neg    
            
            ? magVector = evecs * magVector
            ? xCorr = magVector(1, :)
            ? yCorr = magVector(2, :)
            ? zCorr = magVector(3, :)

'''
import numpy as np
import matplotlib.pyplot as plt
from quaternion import Vector3D
import time
from scipy.integrate import cumtrapz

CALSIZE = 1000

def get_gyr():
    return Vector3D(0,0,0)

def get_acc_mag():
    return Vector3D(0,0,0), Vector3D(0,0,0)

def gyr_bias():
    # read gyroscope data
    print("-"*50)
    print('Gyro Calibrating - Keep the IMU Steady')
    [get_gyr() for ii in range(0,CALSIZE)] # clear buffer before calibration
    gyr_array = []
    gyr_offsets = [0.0,0.0,0.0]
    while True:
        try:
            wx,wy,wz = get_gyr() # get gyro vals
            gyr_array.append([wx,wy,wz])
        except:
            continue


        if np.shape(gyr_array)[0]==CALSiZE:
            for qq in range(0,3):
                gyr_offsets[qq] = np.mean(np.array(gyr_array)[:,qq]) # average
            break
    print('Gyro Calibration Complete')
    print("-"*50)
    return gyr_offsets

def gyr_bias_cal():

    gyr_labels = ['w_x','w_y','w_z'] # gyro labels for plots
    gyr_offsets = gyr_bias_cal() # calculate gyro offsets

    print()
    # Apply offset to new set of data
    data = np.array([get_gyr() for ii in range(0,CALSIZE)]) # new values
      
    # Plot new set of data
    plt.style.use('ggplot')
    fig,axs = plt.subplots(2,1,figsize=(12,9))
    for ii in range(0,3):
        axs[0].plot(data[:,ii], label='${}$, Uncalibrated'.format(gyr_labels[ii]))
        axs[1].plot(data[:,ii]-gyr_offsets[ii], label='${}$, Calibrated'.format(gyr_labels[ii]))
    axs[0].legend(fontsize=14);axs[1].legend(fontsize=14)
    axs[0].set_ylabel('$w_{x,y,z}$ [$^{\circ}/s$]',fontsize=18)
    axs[1].set_ylabel('$w_{x,y,z}$ [$^{\circ}/s$]',fontsize=18)
    axs[1].set_xlabel('Sample',fontsize=18)
    axs[0].set_ylim([-2,2]);axs[1].set_ylim([-2,2])
    axs[0].set_title('Gyroscope Calibration Offset Correction',fontsize=22)
    fig.savefig('gyro_calibration_output.png',dpi=300, bbox_inches='tight',facecolor='#FCFCFC')
    fig.show()

def gyr_drift(offset):

    ##################################
    # Record Data

    input("Press Enter and Rotate Device 360 degrees")
    print("Recording Data...")
    record_time = 5 # how long to record
    data,t_vec = [],[]
    t0 = time.time()
    while (time.time() - t0) < record_time:
        data.append(get_gyr())
        t_vec.append(time.time()-t0)
 
    samp_rate = np.shape(data)[0]/(t_vec[-1]-t_vec[0]) # sample rate
    print("Stopped Recording\nSample Rate: {0:2.0f} Hz".format(samp_rate))

    ##################################
    # Offset and Integration of gyro and plotting results

    data_offseted = np.array(data)[:] - offset
    integ_x = cumtrapz(data_offseted[0,:],x=t_vec) # integrate over time
    integ_y = cumtrapz(data_offseted[1,:],x=t_vec)  
    integ_z = cumtrapz(data_offseted[2,:],x=t_vec) 
    #
    # print out results
    print("Integration of x in {0:2.2f}m".format(integ_x[-1]))
    print("Integration of y in {0:2.2f}m".format(integ_y[-1]))
    print("Integration of z in {0:2.2f}m".format(integ_z[-1]))

    #
    # plotting routine
    plt.style.use('ggplot')
    fig,axs = plt.subplots(2,1,figsize=(12,9))
    axs[0].plot(t_vec,data_offseted,label="$"+gyro_labels[rot_axis]+"$")
    axs[1].plot(t_vec[1:],integ1_array,label=r"$\theta_"+gyro_labels[rot_axis].split("_")[1]+"$")
    [axs[ii].legend(fontsize=16) for ii in range(0,len(axs))]
    axs[0].set_ylabel('Angular Velocity, $\omega_{}$ [$^\circ/s$]'.format(gyro_labels[rot_axis].split("_")[1]),fontsize=16)
    axs[1].set_ylabel(r'Rotation, $\theta_{}$ [$^\circ$]'.format(gyro_labels[rot_axis].split("_")[1]),fontsize=16)
    axs[1].set_xlabel('Time [s]',fontsize=16)
    axs[0].set_title('Gyroscope Integration over 180$^\circ$ Rotation', fontsize=18)
    fig.savefig('gyroscope_integration_180deg_rot.png',dpi=300,
                bbox_inches='tight',facecolor='#FFFFFF')
    plt.show()