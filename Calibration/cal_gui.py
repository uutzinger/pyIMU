"""
cal_gui.py - Calibration GUI for IMU devices

Copyright (C) 2012 Fabio Varesano <fabio at varesano dot net>
Updates by Urs Utzinger 2023: Qt5, Gyroscope, Serial Data Transfer 

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

import sys, os

from PyQT5 import uic
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QMainWindow, QCursor, QFileDialog
from PyQt5.QtCore import Qt, QThread, QSettings, SIGNAL

import pyqtgraph.opengl as gl

import json

import numpy as np
import serial, time
import struct
import cal_lib, numpy
import logging

# User Settings
######################################################################

BAUDRATE = 115200

acc_file_name = "acc.txt"
mag_file_name = "mag.txt"
gyr_file_name = "gyr.txt"

calibration_h_file_name     = "calibration.h"
calibration_json_file_name  = "calibration.JSON"

acc_range = 15   # +/- Display Range is around 10 m/s^2
mag_range = 100  # +/- Display Range is around60 micro Tesla
gyr_range = 10   # +/- Display Range 33rpm = 33*60rps = 33*60*2pi rad/s = 3.49 rad/s

# QT Settings
######################################################################

# Deal with high resolution displays
if hasattr(Qt, 'AA_EnableHighDpiScaling'):
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

# Support Function
######################################################################

def hex_to_float(hex_chars):
    '''Unpack 8 bytes to float'''
    hex_bytes = bytes.fromhex(hex_chars)  # Convert hex characters to bytes
    return struct.unpack('!f', hex_bytes)[0]     

# Main Class
######################################################################

class FreeIMUCal(QMainWindow):

  def __init__(self, parent=None):

    super(FreeIMUCal, self).__init__(parent) # parent constructor

    self.logger = logging.getLogger("Main")

    #----------------------------------------------------------------------------------------------------------------------
    # User Interface
    #----------------------------------------------------------------------------------------------------------------------
    self.ui = uic.loadUi('freeimu_cal.ui', self)
    # window_icon = pkg_resources.resource_filename('camera_gui.images', 'camera_48.png')
    # self.setWindowIcon(QIcon(window_icon))
    self.setWindowTitle("FreeIMU Cal")
    
    # load user settings
    self.settings = QSettings("FreeIMU Calibration Application", "Fabio Varesano")
    # restore previous serial port used
    self.serialPortEdit.setText(self.settings.value("calgui/serialPortEdit", "").toString())
    
    # when user hits enter, we generate the clicked signal to the button so that connection starts
    self.connect(self.serialPortEdit, SIGNAL("returnPressed()"), self.connectButton, SIGNAL("clicked()"))
    
    # Connect up the buttons to their functions
    self.connectButton.clicked.connect(self.serial_connect)
    self.samplingToggleButton.clicked.connect(self.sampling_start)
    self.set_status("Disconnected")
    
    # data storages
    self.acc_data = numpy.empty([1,3])
    self.mag_data = numpy.empty([1,3])
    self.gyr_data = numpy.empty([1,3])
    
    # setup graphs
    self.ui.accXY.setXRange(-acc_range, acc_range)
    self.ui.accXY.setYRange(-acc_range, acc_range)
    self.ui.accYZ.setXRange(-acc_range, acc_range)
    self.ui.accYZ.setYRange(-acc_range, acc_range)
    self.ui.accZX.setXRange(-acc_range, acc_range)
    self.ui.accZX.setYRange(-acc_range, acc_range)
    
    self.ui.accXY.setAspectLocked()
    self.ui.accYZ.setAspectLocked()
    self.ui.accZX.setAspectLocked()

    self.ui.gyrXY.setXRange(-gyr_range, gyr_range)
    self.ui.gyrXY.setYRange(-gyr_range, gyr_range)
    self.ui.gyrYZ.setXRange(-gyr_range, gyr_range)
    self.ui.gyrYZ.setYRange(-gyr_range, gyr_range)
    self.ui.gyrZX.setXRange(-gyr_range, gyr_range)
    self.ui.gyrZX.setYRange(-gyr_range, gyr_range)
    
    self.ui.gyrXY.setAspectLocked()
    self.ui.gyrYZ.setAspectLocked()
    self.ui.gyrZX.setAspectLocked()
    
    self.ui.magXY.setXRange(-mag_range, mag_range)
    self.ui.magXY.setYRange(-mag_range, mag_range)
    self.ui.magYZ.setXRange(-mag_range, mag_range)
    self.ui.magYZ.setYRange(-mag_range, mag_range)
    self.ui.magZX.setXRange(-mag_range, mag_range)
    self.ui.magZX.setYRange(-mag_range, mag_range)
    
    self.ui.magXY.setAspectLocked()
    self.ui.magYZ.setAspectLocked()
    self.ui.magZX.setAspectLocked()
    
    self.ui.accXY_cal.setXRange(-1.5, 1.5)
    self.ui.accXY_cal.setYRange(-1.5, 1.5)
    self.ui.accYZ_cal.setXRange(-1.5, 1.5)
    self.ui.accYZ_cal.setYRange(-1.5, 1.5)
    self.ui.accZX_cal.setXRange(-1.5, 1.5)
    self.ui.accZX_cal.setYRange(-1.5, 1.5)
    
    self.ui.accXY_cal.setAspectLocked()
    self.ui.accYZ_cal.setAspectLocked()
    self.ui.accZX_cal.setAspectLocked()

    self.ui.gyrXY_cal.setXRange(-1.5, 1.5)
    self.ui.gyrXY_cal.setYRange(-1.5, 1.5)
    self.ui.gyrYZ_cal.setXRange(-1.5, 1.5)
    self.ui.gyrYZ_cal.setYRange(-1.5, 1.5)
    self.ui.gyrZX_cal.setXRange(-1.5, 1.5)
    self.ui.gyrZX_cal.setYRange(-1.5, 1.5)
    
    self.ui.gyrXY_cal.setAspectLocked()
    self.ui.gyrYZ_cal.setAspectLocked()
    self.ui.gyrZX_cal.setAspectLocked()
    
    self.ui.magXY_cal.setXRange(-1.5, 1.5)
    self.ui.magXY_cal.setYRange(-1.5, 1.5)
    self.ui.magYZ_cal.setXRange(-1.5, 1.5)
    self.ui.magYZ_cal.setYRange(-1.5, 1.5)
    self.ui.magZX_cal.setXRange(-1.5, 1.5)
    self.ui.magZX_cal.setYRange(-1.5, 1.5)
    
    self.ui.magXY_cal.setAspectLocked()
    self.ui.magYZ_cal.setAspectLocked()
    self.ui.magZX_cal.setAspectLocked()
    
    self.ui.acc3D.opts['distance'] = 30000
    self.ui.acc3D.show()

    self.ui.gyr3D.opts['distance'] = 30000
    self.ui.gyr3D.show()
    
    self.ui.mag3D.opts['distance'] = 2000
    self.ui.mag3D.show()
    
    ax = gl.GLAxisItem()
    ax.setSize(x=20000, y=20000, z=20000)
    self.ui.acc3D.addItem(ax)
    
    gx = gl.GLAxisItem()
    gx.setSize(x=20000, y=20000, z=20000)
    self.ui.gyr3D.addItem(gx)
    
    mx = gl.GLAxisItem()
    mx.setSize(x=1000, y=1000, z=1000)
    self.ui.mag3D.addItem(ax)
    
    self.acc3D_sp = gl.GLScatterPlotItem()
    self.ui.acc3D.addItem(self.acc3D_sp)

    self.gyr3D_sp = gl.GLScatterPlotItem()
    self.ui.gyr3D.addItem(self.gyr3D_sp)
    
    self.mag3D_sp = gl.GLScatterPlotItem()
    self.ui.mag3D.addItem(self.mag3D_sp)
    
    # axis for the cal 3D graph
    g_a = gl.GLAxisItem()
    g_a.setSize(x=10000, y=10000, z=10000)
    self.ui.acc3D_cal.addItem(g_a)

    g_g = gl.GLAxisItem()
    g_g.setSize(x=10000, y=10000, z=10000)
    self.ui.gyr3D_cal.addItem(g_g)

    g_m = gl.GLAxisItem()
    g_m.setSize(x=1000, y=1000, z=1000)
    self.ui.mag3D_cal.addItem(g_m)
    
  def set_status(self, status):
    self.statusbar.showMessage(self.tr(status))

  def serial_connect(self):
    self.serial_port = str(self.serialPortEdit.text())
    # save serial value to user settings
    self.settings.setValue("calgui/serialPortEdit", self.serial_port)
    
    self.connectButton.setEnabled(False)
    # waiting mouse cursor
    QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
    self.set_status("Connecting to " + self.serial_port + " ...")
    
    # TODO: serial port field input validation!
    
    try:
      self.ser = serial.Serial(
        port= self.serial_port,
        baudrate=BAUDRATE,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS
      )
      
      if self.ser.isOpen():
        print("Serial port opened correctly")
        self.set_status("Connection Successful. Waiting for device reset...")

        # wait for device reset on serial open
        time.sleep(3)
        
        self.ser.write('v') # ask version
        self.set_status("Connected to: " + self.ser.readline()) # TODO: hangs if a wrong serial protocol has been loaded. To be fixed.
        
        self.connectButton.setText("Disconnect")
        self.connectButton.clicked.connect(self.serial_disconnect)
        self.serialPortEdit.setEnabled(False)
        self.serialProtocol.setEnabled(False)
        
        self.samplingToggleButton.setEnabled(True)
        
        self.clearCalibrationEEPROMButton.setEnabled(True)
        self.clearCalibrationEEPROMButton.clicked.connect(self.clear_calibration_eeprom)
        
    except serial.serialutil.SerialException as e:
      self.connectButton.setEnabled(True)
      self.set_status("Could not connect: " + str(e))
      
    # restore mouse cursor
    QApplication.restoreOverrideCursor()
    self.connectButton.setEnabled(True)

    
  def serial_disconnect(self):
    print("Disconnecting from " + self.serial_port)
    self.ser.close()
    self.set_status("Disconnected")
    self.serialPortEdit.setEnabled(True)
    self.serialProtocol.setEnabled(True)
    
    self.connectButton.setText("Connect")
    self.connectButton.clicked.disconnect(self.serial_disconnect)
    self.connectButton.clicked.connect(self.serial_connect)
    
    self.samplingToggleButton.setEnabled(False)
    
    self.clearCalibrationEEPROMButton.setEnabled(False)
    self.clearCalibrationEEPROMButton.clicked.disconnect(self.clear_calibration_eeprom)
      
  def sampling_start(self):
    self.serWorker = SerialWorker(ser = self.ser)
    self.connect(self.serWorker, SIGNAL("new_data(PyQt_PyObject)"), self.newData)
    self.serWorker.start()
    print("Starting SerialWorker")
    self.samplingToggleButton.setText("Stop Sampling")
    
    self.samplingToggleButton.clicked.disconnect(self.sampling_start)
    self.samplingToggleButton.clicked.connect(self.sampling_end)
    
  def sampling_end(self):
    self.serWorker.exiting = True
    self.serWorker.quit()
    self.serWorker.wait()
    self.samplingToggleButton.setText("Start Sampling")
    self.samplingToggleButton.clicked.disconnect(self.sampling_end)
    self.samplingToggleButton.clicked.connect(self.sampling_start)
    
    self.calibrateButton.setEnabled(True)
    self.calAlgorithmComboBox.setEnabled(True)
    self.calibrateButton.clicked.connect(self.calibrate)
  
  def calibrate(self):
    # read file and run calibration algorithm
    (self.acc_offset, self.acc_correctionMat) = cal_lib.calibrate_from_file(acc_file_name)
    (self.gyr_offset, self.gyr_correctionMat) = cal_lib.calibrate_from_file(gyr_file_name)
    (self.mag_offset, self.mag_correctionMat) = cal_lib.calibrate_from_file(mag_file_name)
    self.acc_scale = np.diag(self.acc_correctionMat)
    self.gyr_scale = np.diag(self.gyr_correctionMat)
    self.mag_scale = np.diag(self.mag_correctionMat)
    # show calibrated tab
    self.tabWidget.setCurrentIndex(1)
    
    #populate acc calibration output on gui
    self.calRes_acc_OSx.setText(str(self.acc_offset[0]))
    self.calRes_acc_OSy.setText(str(self.acc_offset[1]))
    self.calRes_acc_OSz.setText(str(self.acc_offset[2]))
    
    self.calRes_acc_SCx.setText(str(self.acc_scale[0]))
    self.calRes_acc_SCy.setText(str(self.acc_scale[1]))
    self.calRes_acc_SCz.setText(str(self.acc_scale[2]))

    #populate gyr calibration output on gui
    self.calRes_gyr_OSx.setText(str(self.gyr_offset[0]))
    self.calRes_gyr_OSy.setText(str(self.gyr_offset[1]))
    self.calRes_gyr_OSz.setText(str(self.gyr_offset[2]))
    
    self.calRes_gyr_SCx.setText(str(self.gyr_scale[0]))
    self.calRes_gyr_SCy.setText(str(self.gyr_scale[1]))
    self.calRes_gyr_SCz.setText(str(self.gyr_scale[2]))
    
    #populate mag calibration output on gui
    self.calRes_mag_OSx.setText(str(self.mag_offset[0]))
    self.calRes_mag_OSy.setText(str(self.mag_offset[1]))
    self.calRes_mag_OSz.setText(str(self.mag_offset[2]))
    
    self.calRes_mag_SCx.setText(str(self.mag_scale[0]))
    self.calRes_mag_SCy.setText(str(self.mag_scale[1]))
    self.calRes_mag_SCz.setText(str(self.mag_scale[2]))
    
    # compute calibrated data
    self.acc_cal_data = cal_lib.compute_calibrate_data(self.acc_data, self.acc_offset, self.acc_correctionMat)
    self.gyr_cal_data = cal_lib.compute_calibrate_data(self.gyr_data, self.gyr_offset, self.gyr_correctionMat)
    self.mag_cal_data = cal_lib.compute_calibrate_data(self.mag_data, self.mag_offset, self.mag_correctionMat)
    
    # populate 2D graphs with calibrated data
    self.accXY_cal.plot(x = self.acc_cal_data[:,0], y = self.acc_cal_data[:,1], clear = True, pen='r')
    self.accYZ_cal.plot(x = self.acc_cal_data[:,1], y = self.acc_cal_data[:,2], clear = True, pen='g')
    self.accZX_cal.plot(x = self.acc_cal_data[:,2], y = self.acc_cal_data[:,0], clear = True, pen='b')

    self.gyrXY_cal.plot(x = self.gyr_cal_data[:,0], y = self.gyr_cal_data[:,1], clear = True, pen='r')
    self.gyrYZ_cal.plot(x = self.gyr_cal_data[:,1], y = self.gyr_cal_data[:,2], clear = True, pen='g')
    self.gyrZX_cal.plot(x = self.gyr_cal_data[:,2], y = self.gyr_cal_data[:,0], clear = True, pen='b')
    
    self.magXY_cal.plot(x = self.mag_cal_data[:,0], y = self.mag_cal_data[:,1], clear = True, pen='r')
    self.magYZ_cal.plot(x = self.mag_cal_data[:,1], y = self.mag_cal_data[:,2], clear = True, pen='g')
    self.magZX_cal.plot(x = self.mag_cal_data[:,2], y = self.mag_cal_data[:,0], clear = True, pen='b')
    
    # populate 3D graphs with calibrated data
    
    sp = gl.GLScatterPlotItem(pos=self.acc_cal_data, color = (1, 1, 1, 1), size=2)
    self.acc3D_cal.addItem(sp)

    sp = gl.GLScatterPlotItem(pos=self.gyr_cal_data, color = (1, 1, 1, 1), size=2)
    self.gyr3D_cal.addItem(sp)
    
    sp = gl.GLScatterPlotItem(pos=self.mag_cal_data, color = (1, 1, 1, 1), size=2)
    self.mag3D_cal.addItem(sp)
    
    #enable calibration buttons to activate calibration storing functions
    self.saveCalibrationHeaderButton.setEnabled(True)
    self.saveCalibrationHeaderButton.clicked.connect(self.save_calibration_header)

    self.saveCalibrationJSONButton.setEnabled(True)
    self.saveCalibrationJSONButton.clicked.connect(self.save_calibration_json)
    
    self.saveCalibrationEEPROMButton.setEnabled(True)
    self.saveCalibrationEEPROMButton.clicked.connect(self.save_calibration_eeprom)
    
  def save_calibration_json(self):
    data = {
      "acc_offset_x": self.acc_offset[0], 
      "acc_offset_y": self.acc_offset[1], 
      "acc_offset_z": self.acc_offset[2], 
      "acc_scale_x":  self.acc_scale[0], 
      "acc_scale_y":  self.acc_scale[1], 
      "acc_scale_z":  self.acc_scale[2], 
      "acc_cMat_00":  self.acc_correctionMat[0,0],
      "acc_cMat_01":  self.acc_correctionMat[0,1],
      "acc_cMat_02":  self.acc_correctionMat[0,2],
      "acc_cMat_10":  self.acc_correctionMat[1,0],
      "acc_cMat_11":  self.acc_correctionMat[1,1],
      "acc_cMat_12":  self.acc_correctionMat[1,2],
      "acc_cMat_20":  self.acc_correctionMat[2,0],
      "acc_cMat_21":  self.acc_correctionMat[2,1],
      "acc_cMat_22":  self.acc_correctionMat[2,2],

      "gyr_offset_x": self.gyr_offset[0], 
      "gyr_offset_y": self.gyr_offset[1], 
      "gyr_offset_z": self.gyr_offset[2], 
      "gyr_scale_x":  self.gyr_scale[0], 
      "gyr_scale_y":  self.gyr_scale[1], 
      "gyr_scale_z":  self.gyr_scale[2], 
      "gyr_cMat_00":  self.gyr_correctionMat[0,0],
      "gyr_cMat_01":  self.gyr_correctionMat[0,1],
      "gyr_cMat_02":  self.gyr_correctionMat[0,2],
      "gyr_cMat_10":  self.gyr_correctionMat[1,0],
      "gyr_cMat_11":  self.gyr_correctionMat[1,1],
      "gyr_cMat_12":  self.gyr_correctionMat[1,2],
      "gyr_cMat_20":  self.gyr_correctionMat[2,0],
      "gyr_cMat_21":  self.gyr_correctionMat[2,1],
      "gyr_cMat_22":  self.gyr_correctionMat[2,2],

      "mag_offset_x": self.mag_offset[0], 
      "mag_offset_y": self.mag_offset[1], 
      "mag_offset_z": self.mag_offset[2], 
      "mag_scale_x":  self.mag_scale[0], 
      "mag_scale_x":  self.mag_scale[1], 
      "mag_scale_x":  self.mag_scale[2],
      "mag_cMat_00":  self.mag_correctionMat[0,0],
      "mag_cMat_01":  self.mag_correctionMat[0,1],
      "mag_cMat_02":  self.mag_correctionMat[0,2],
      "mag_cMat_10":  self.mag_correctionMat[1,0],
      "mag_cMat_11":  self.mag_correctionMat[1,1],
      "mag_cMat_12":  self.mag_correctionMat[1,2],
      "mag_cMat_20":  self.mag_correctionMat[2,0],
      "mag_cMat_21":  self.mag_correctionMat[2,1],
      "mag_cMat_22":  self.mag_correctionMat[2,2]
      
    }
    
    calibration_h_folder = QFileDialog.getExistingDirectory(self, "Select the Folder to which save the calibration.h file")
    with open(os.path.join(str(calibration_h_folder), calibration_json_file_name), "w") as file:
      json.dump(data, file)
    
    self.set_status("Calibration saved to: " + str(calibration_h_folder) + calibration_json_file_name + ".\n")
    
    
  def save_calibration_header(self):
    text = """

/**
 * FreeIMU calibration header. Automatically generated by FreeIMU_GUI.
 * Do not edit manually unless you know what you are doing.
*/

#define CALIBRATION_H

const int acc_off_x = %d;
const int acc_off_y = %d;
const int acc_off_z = %d;
const float acc_scale_x = %f;
const float acc_scale_y = %f;
const float acc_scale_z = %f;

const int mag_off_x = %d;
const int mag_off_y = %d;
const int mag_off_z = %d;
const float mag_scale_x = %f;
const float mag_scale_y = %f;
const float mag_scale_z = %f;
"""
    calibration_h_text = text % (self.acc_offset[0], self.acc_offset[1], self.acc_offset[2], self.acc_scale[0], self.acc_scale[1], self.acc_scale[2], self.mag_offset[0], self.mag_offset[1], self.mag_offset[2], self.mag_scale[0], self.mag_scale[1], self.mag_scale[2])
    
    calibration_h_folder = QFileDialog.getExistingDirectory(self, "Select the Folder to which save the calibration.h file")
    calibration_h_file = open(os.path.join(str(calibration_h_folder), calibration_h_file_name), "w")
    calibration_h_file.write(calibration_h_text)
    calibration_h_file.close()
    
    self.set_status("Calibration saved to: " + str(calibration_h_folder) + calibration_h_file_name + " .\nRecompile and upload the program using the FreeIMU library to your microcontroller.")
  
  def save_calibration_eeprom(self):
    self.ser.write("c")
    # pack data into a string 
    offsets = struct.pack('<hhhhhh', self.acc_offset[0], self.acc_offset[1], self.acc_offset[2], self.mag_offset[0], self.mag_offset[1], self.mag_offset[2])
    scales  = struct.pack('<ffffff', self.acc_scale[0], self.acc_scale[1], self.acc_scale[2], self.mag_scale[0], self.mag_scale[1], self.mag_scale[2])
    # transmit to microcontroller
    self.ser.write(offsets)
    self.ser.write(scales)
    self.set_status("Calibration saved to microcontroller EEPROM.")
    # debug written values to console
    print("Calibration values read back from EEPROM:")
    self.ser.write("C")
    for i in range(4):
      print(self.ser.readline())
      
  def clear_calibration_eeprom(self):
    self.ser.write("x")
    # no feedback expected. we assume success.
    self.set_status("Calibration cleared from microcontroller EEPROM.")
    
  def newData(self, reading):
    
    # only display last reading from burst
    if self.ui.accDisplay.checkbox.isChecked():
      np.stack((self.acc_data.stack, np.array([reading[0],reading[1],reading[2]])), axis=0)

    if self.ui.gyrDisplay.checkbox.isChecked():
      np.stack((self.gyr_data.stack, np.array([reading[3],reading[4],reading[5]])), axis=0)
    
    if self.ui.magDisplay.checkbox.isChecked():
      np.stack((self.mag_data.stack, np.array([reading[6],reading[7],reading[8]])), axis=0)
    
    self.accXY.plot(x = self.acc_data[:,0], y = self.acc_data[:,1], clear = True, pen='r')
    self.accYZ.plot(x = self.acc_data[:,1], y = self.acc_data[:,2], clear = True, pen='g')
    self.accZX.plot(x = self.acc_data[:,2], y = self.acc_data[:,0], clear = True, pen='b')

    self.gyrXY.plot(x = self.gyr_data[:,0], y = self.gyr_data[:,1], clear = True, pen='r')
    self.gyrYZ.plot(x = self.gyr_data[:,1], y = self.gyr_data[:,2], clear = True, pen='g')
    self.gyrZX.plot(x = self.gyr_data[:,2], y = self.gyr_data[:,0], clear = True, pen='b')
    
    self.magXY.plot(x = self.mag_data[:,0], y = self.mag_data[:,1], clear = True, pen='r')
    self.magYZ.plot(x = self.mag_data[:,1], y = self.mag_data[:,2], clear = True, pen='g')
    self.magZX.plot(x = self.mag_data[:,2], y = self.mag_data[:,0], clear = True, pen='b')
    
    self.acc3D_sp.setData(pos=self.acc_data, color = (1, 1, 1, 1), size=2)
    self.gyr3D_sp.setData(pos=self.gyr_data, color = (1, 1, 1, 1), size=2)
    self.mag3D_sp.setData(pos=self.mag_data, color = (1, 1, 1, 1), size=2)

class SerialWorker(QThread):
  def __init__(self, parent = None, ser = None):
    QThread.__init__(self, parent)
    self.exiting = False
    self.ser = ser
        
  def run(self):
    print("Starting sampling...")
    if self.ui.accRecord.checkbox.isChecked():
      if self.ui.accAppend.checkbox.isChecked():
        self.acc_file = open(acc_file_name, 'a')
      else:
        self.acc_file = open(acc_file_name, 'w')
    else:
      self.acc_file = None
    if self.ui.gyrRecord.checkbox.isChecked():
      if self.ui.gyrAppend.checkbox.isChecked():
        self.gyr_file = open(gyr_file_name, 'a')
      else:
        self.gyr_file = open(gyr_file_name, 'w')
    else:
      self.gyr_file = None
    if self.ui.magRecord.checkbox.isChecked():
      if self.ui.magAppend.checkbox.isChecked():
        self.mag_file = open(mag_file_name, 'a')
      else:
        self.mag_file = open(mag_file_name, 'w')
    else:
      self.mag_file = None
    
    count = 100 # read 100 values then pass single one to GUI
    in_values = 9 # 3 values for acc, gyr and mag
    reading = [0.0 for i in range(in_values)]
    
    while not self.exiting:
      self.ser.write('b') # request data
      self.ser.write(chr(count))
      for j in range(count):
        for i in range(in_values):
          byte_array = self.ser.read(8)            # byte array of 8 bytes
          hex_chars = ''.join(byte_array.decode()) # convert byte array to string
          reading[i] = hex_to_float(hex_chars)     #

        self.ser.read(2) # consumes remaining '\r\n'

        # prepare readings to store in file

        if self.acc_file != None: 
          acc_readings_line = "{:f} {:f} {:f}\r\n".format(reading[0], reading[1], reading[2])
          self.acc_file.write(acc_readings_line)

        if self.gyr_file != None: 
          gyr_readings_line = "{:f} {:f} {:f}\r\n".format(reading[3], reading[4], reading[5])
          self.gyr_file.write(gyr_readings_line)

        if self.mag_file != None: 
          mag_readings_line = "{:f} {:f} {:f}\r\n".format(reading[6], reading[7], reading[8])
          self.mag_file.write(mag_readings_line)
        
      # every count times we pass some data to the GUI
      self.emit(SIGNAL("new_data(PyQt_PyObject)"), reading)
      print(".")

    # closing acc,gyr and mag files
    if self.acc_file != None: self.acc_file.close()
    if self.gyr_file != None: self.gyr_file.close()
    if self.mag_file != None: self.mag_file.close()
    return 
  
  def __del__(self):
    self.exiting = True
    self.wait()
    print("SerialWorker exits...")


######################################################################
# Main Program
######################################################################

app = QApplication(sys.argv)
window = FreeIMUCal()

window.show()
sys.exit(app.exec_())