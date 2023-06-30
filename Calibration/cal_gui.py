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
from PyQt5 import uic
from PyQt5.QtCore import Qt, QObject, QThread, QSettings, pyqtSignal
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QFileDialog
from PyQt5.QtGui import QCursor, QIcon

import pyqtgraph.opengl as gl
from pyqtgraph import PlotWidget
from pyqtgraph.opengl import GLViewWidget

import sys, os
import numpy as np
import serial, time
import struct
import json
import logging
import pathlib

import cal_lib

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

######################################################################
# Serial Worker Class
######################################################################

class SerialWorker(QThread):
  
  new_data = pyqtSignal(object)
  
  def __init__(self, ser, ui):
    QThread.__init__(self)
    self.ser = ser
    self.ui = ui
    self.exiting = False
        
  def run(self):
    print("Setting up sampling...")
    if self.ui.accRecord.isChecked():
      if self.ui.accAppend.isChecked():
        self.acc_file = open(acc_file_name, 'a')
      else:
        self.acc_file = open(acc_file_name, 'w')
    else:
      self.acc_file = None
    if self.ui.gyrRecord.isChecked():
      if self.ui.gyrAppend.isChecked():
        self.gyr_file = open(gyr_file_name, 'a')
      else:
        self.gyr_file = open(gyr_file_name, 'w')
    else:
      self.gyr_file = None
    if self.ui.magRecord.isChecked():
      if self.ui.magAppend.isChecked():
        self.mag_file = open(mag_file_name, 'a')
      else:
        self.mag_file = open(mag_file_name, 'w')
    else:
      self.mag_file = None
    
    count = 100 # read 100 values then pass last one to GUI
    in_values = 9 # 3 values for acc, gyr and mag
    readings = [0.0 for i in range(in_values)]

    print("Start sampling...")    
    while not self.exiting:
      self.ser.flushInput()                         # clear serial input buffer
      self.ser.write( ("b{}\r\n".format(count)).encode())    # request data
      for j in range(count):  
        for i in range(in_values):
          byte_array = self.ser.read(8)             # byte array of 8 bytes for each floating point value (float is 4 bytes and when converted to readable hex is 8 bytes)
          hex_chars = ''.join(byte_array.decode())  # convert byte array to string
          readings[i] = hex_to_float(hex_chars)     #
        self.ser.read(2)                            # consumes remaining '\r\n'

        # store readings in files
        if self.acc_file != None: 
          acc_readings_line = '{:f} {:f} {:f}\r\n'.format(readings[0], readings[1], readings[2])
          self.acc_file.write(acc_readings_line)
        if self.gyr_file != None: 
          gyr_readings_line = '{:f} {:f} {:f}\r\n'.format(readings[3], readings[4], readings[5])
          self.gyr_file.write(gyr_readings_line)
        if self.mag_file != None: 
          mag_readings_line = '{:f} {:f} {:f}\r\n'.format(readings[6], readings[7], readings[8])
          self.mag_file.write(mag_readings_line)
        
      # every count times we pass last reading to the GUI
      self.new_data.emit(readings)      
      print('.')

    # closing acc,gyr and mag files
    if self.acc_file != None: self.acc_file.close()
    if self.gyr_file != None: self.gyr_file.close()
    if self.mag_file != None: self.mag_file.close()
    return 
  
  def __del__(self):
    self.exiting = True
    self.wait()
    print('SerialWorker exits...')

######################################################################
# Main Program
######################################################################

class FreeIMUCal(QMainWindow):

  def __init__(self):
    super().__init__()

    self.logger = logging.getLogger('Main')

    # Load UI and setup widgets
    self.ui = uic.loadUi('freeimu_cal.ui', self)
    
    self.setWindowTitle('FreeIMU Cal')
    current_directory = str(pathlib.Path(__file__).parent.absolute())
    path = current_directory + '/FreeIMU.png'
    self.setWindowIcon(QIcon(path))
    
    # load user settings
    self.settings = QSettings('FreeIMU Calibration Application', 'Fabio Varesano')
    # restore previous serial port used
    # self.ui.serialPortEdit.setText(self.settings.value('calgui/serialPortEdit', '').toString())
    self.ui.serialPortEdit.setText(self.settings.value('calgui/serialPortEdit', ''))
    
    # when user hits enter, we generate the clicked signal to the button so that connection starts
    self.ui.serialPortEdit.returnPressed.connect(self.ui.connectButton.click)

    # Connect up the buttons to their functions
    self.ui.connectButton.clicked.connect(self.serial_connect)
    self.ui.samplingToggleButton.clicked.connect(self.sampling_start)
    self.set_status('Disconnected')

    # Setup graphs
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
    
    # Axis for the cal 3D graph
    g_a = gl.GLAxisItem()
    g_a.setSize(x=10000, y=10000, z=10000)
    self.ui.acc3D_cal.addItem(g_a)

    g_g = gl.GLAxisItem()
    g_g.setSize(x=10000, y=10000, z=10000)
    self.ui.gyr3D_cal.addItem(g_g)

    g_m = gl.GLAxisItem()
    g_m.setSize(x=1000, y=1000, z=1000)
    self.ui.mag3D_cal.addItem(g_m)

    # data storages
    self.acc_data = np.empty([1,3])
    self.mag_data = np.empty([1,3])
    self.gyr_data = np.empty([1,3])
      
  def set_status(self, status):
    self.ui.statusbar.showMessage(self.tr(status))

  def serial_connect(self):
    self.serial_port = str(self.serialPortEdit.text())
    # save serial value to user settings
    self.settings.setValue('calgui/serialPortEdit', self.serial_port)
    
    self.connectButton.setEnabled(False)
    # waiting mouse cursor
    QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
    self.set_status('Connecting to ' + self.serial_port + ' ...')
        
    try:


      self.ser = serial.Serial(
        port      = self.serial_port,
        baudrate  = BAUDRATE,
        parity    = serial.PARITY_NONE,
        stopbits  = serial.STOPBITS_ONE,
        bytesize  = serial.EIGHTBITS,
        timeout   = 5.,
        write_timeout = 5.
      )
      
      if self.ser.isOpen():
        print('Serial port is open')
        self.set_status('Serial Port Opened. Waiting for Device Response...')

        # wait for device reset on serial open
        time.sleep(3)
        
        is_connected = False      
        msg_out= b'v\r\n'
        while is_connected == False:
          try:
            err = self.ser.write(msg_out) # ask version
            time.sleep(0.5)
            msg_in = self.ser.read_until(b'\n')
            line_in = msg_in.decode().strip()
            if len(line_in) > 0:
              self.set_status('Connected to: ' + line_in)
              is_connected = True
            else:
              time.sleep(1)
          except: 
            time.sleep(1)
        
        self.ui.connectButton.setText('Disconnect')
        self.ui.connectButton.clicked.connect(self.serial_disconnect)
        self.ui.serialPortEdit.setEnabled(False)
        self.ui.serialProtocol.setEnabled(False)
        
        self.ui.samplingToggleButton.setEnabled(True)
        
        # If you have device with EEPROM
        self.ui.clearCalibrationEEPROMButton.setEnabled(True)
        self.ui.clearCalibrationEEPROMButton.clicked.connect(self.clear_calibration_eeprom)
        
    except serial.serialutil.SerialException as e:
      self.ui.connectButton.setEnabled(True)
      self.ui.set_status('Could not connect: ' + str(e))
      
    # restore mouse cursor
    QApplication.restoreOverrideCursor()
    self.ui.connectButton.setEnabled(True)

  def serial_disconnect(self):
    print('Disconnecting from ' + self.serial_port)
    self.ser.close()
    self.ui.set_status('Disconnected')
    self.ui.serialPortEdit.setEnabled(True)
    self.ui.serialProtocol.setEnabled(True)
    
    self.ui.connectButton.setText('Connect')
    self.ui.connectButton.clicked.disconnect(self.serial_disconnect)
    self.ui.connectButton.clicked.connect(self.serial_connect)
    
    self.ui.samplingToggleButton.setEnabled(False)
    
    self.ui.clearCalibrationEEPROMButton.setEnabled(False)
    self.ui.clearCalibrationEEPROMButton.clicked.disconnect(self.clear_calibration_eeprom)
      
  def sampling_start(self):
    self.serWorker = SerialWorker(ser = self.ser, ui = self.ui)
    self.serWorker.new_data.connect(self.newData)
    self.serWorker.start()
    print('Starting SerialWorker')
    self.ui.samplingToggleButton.setText('Stop Sampling')
    self.ui.samplingToggleButton.clicked.disconnect(self.sampling_start)
    self.ui.samplingToggleButton.clicked.connect(self.sampling_end)
    
  def sampling_end(self):
    self.serWorker.exiting = True
    self.serWorker.quit()
    self.serWorker.wait()
    self.ui.samplingToggleButton.setText('Start Sampling')
    self.ui.samplingToggleButton.clicked.disconnect(self.sampling_end)
    self.ui.samplingToggleButton.clicked.connect(self.sampling_start)
    
    self.ui.calibrateButton.setEnabled(True)
    self.ui.calAlgorithmComboBox.setEnabled(True)
    self.ui.calibrateButton.clicked.connect(self.calibrate)
  
  def calibrate(self):
    # read file and run calibration algorithm
    (self.acc_offset, self.acc_correctionMat) = cal_lib.calibrate_from_file(acc_file_name)
    (self.gyr_offset, self.gyr_correctionMat) = cal_lib.calibrate_from_file(gyr_file_name)
    (self.mag_offset, self.mag_correctionMat) = cal_lib.calibrate_from_file(mag_file_name)
    self.acc_scale = np.diag(self.acc_correctionMat)
    self.gyr_scale = np.diag(self.gyr_correctionMat)
    self.mag_scale = np.diag(self.mag_correctionMat)
    # show calibrated tab
    self.ui.tabWidget.setCurrentIndex(1)
    
    #populate acc calibration output on gui
    self.ui.calRes_acc_OSx.setText(str(self.acc_offset[0]))
    self.ui.calRes_acc_OSy.setText(str(self.acc_offset[1]))
    self.ui.calRes_acc_OSz.setText(str(self.acc_offset[2]))
    
    self.ui.calRes_acc_SCx.setText(str(self.acc_scale[0]))
    self.ui.calRes_acc_SCy.setText(str(self.acc_scale[1]))
    self.ui.calRes_acc_SCz.setText(str(self.acc_scale[2]))

    #populate gyr calibration output on gui
    self.ui.calRes_gyr_OSx.setText(str(self.gyr_offset[0]))
    self.ui.calRes_gyr_OSy.setText(str(self.gyr_offset[1]))
    self.ui.calRes_gyr_OSz.setText(str(self.gyr_offset[2]))
    
    self.ui.calRes_gyr_SCx.setText(str(self.gyr_scale[0]))
    self.ui.calRes_gyr_SCy.setText(str(self.gyr_scale[1]))
    self.ui.calRes_gyr_SCz.setText(str(self.gyr_scale[2]))
    
    #populate mag calibration output on gui
    self.ui.calRes_mag_OSx.setText(str(self.mag_offset[0]))
    self.ui.calRes_mag_OSy.setText(str(self.mag_offset[1]))
    self.ui.calRes_mag_OSz.setText(str(self.mag_offset[2]))
    
    self.ui.calRes_mag_SCx.setText(str(self.mag_scale[0]))
    self.ui.calRes_mag_SCy.setText(str(self.mag_scale[1]))
    self.ui.calRes_mag_SCz.setText(str(self.mag_scale[2]))
    
    # compute calibrated data
    self.acc_cal_data = cal_lib.compute_calibrate_data(self.acc_data, self.acc_offset, self.acc_correctionMat)
    self.gyr_cal_data = cal_lib.compute_calibrate_data(self.gyr_data, self.gyr_offset, self.gyr_correctionMat)
    self.mag_cal_data = cal_lib.compute_calibrate_data(self.mag_data, self.mag_offset, self.mag_correctionMat)
    
    # populate 2D graphs with calibrated data
    self.ui.accXY_cal.plot(x = self.acc_cal_data[:,0], y = self.acc_cal_data[:,1], clear = True, pen='r')
    self.ui.accYZ_cal.plot(x = self.acc_cal_data[:,1], y = self.acc_cal_data[:,2], clear = True, pen='g')
    self.ui.accZX_cal.plot(x = self.acc_cal_data[:,2], y = self.acc_cal_data[:,0], clear = True, pen='b')

    self.ui.gyrXY_cal.plot(x = self.gyr_cal_data[:,0], y = self.gyr_cal_data[:,1], clear = True, pen='r')
    self.ui.gyrYZ_cal.plot(x = self.gyr_cal_data[:,1], y = self.gyr_cal_data[:,2], clear = True, pen='g')
    self.ui.gyrZX_cal.plot(x = self.gyr_cal_data[:,2], y = self.gyr_cal_data[:,0], clear = True, pen='b')
    
    self.ui.magXY_cal.plot(x = self.mag_cal_data[:,0], y = self.mag_cal_data[:,1], clear = True, pen='r')
    self.ui.magYZ_cal.plot(x = self.mag_cal_data[:,1], y = self.mag_cal_data[:,2], clear = True, pen='g')
    self.ui.magZX_cal.plot(x = self.mag_cal_data[:,2], y = self.mag_cal_data[:,0], clear = True, pen='b')
    
    # populate 3D graphs with calibrated data
    
    sp = gl.GLScatterPlotItem(pos=self.acc_cal_data, color = (1, 1, 1, 1), size=2)
    self.ui.acc3D_cal.addItem(sp)

    sp = gl.GLScatterPlotItem(pos=self.gyr_cal_data, color = (1, 1, 1, 1), size=2)
    self.ui.gyr3D_cal.addItem(sp)
    
    sp = gl.GLScatterPlotItem(pos=self.mag_cal_data, color = (1, 1, 1, 1), size=2)
    self.ui.mag3D_cal.addItem(sp)
    
    #enable calibration buttons to activate calibration storing functions
    self.ui.saveCalibrationHeaderButton.setEnabled(True)
    self.ui.saveCalibrationHeaderButton.clicked.connect(self.save_calibration_header)

    self.ui.saveCalibrationJSONButton.setEnabled(True)
    self.ui.saveCalibrationJSONButton.clicked.connect(self.save_calibration_json)
    
    self.ui.saveCalibrationEEPROMButton.setEnabled(True)
    self.ui.saveCalibrationEEPROMButton.clicked.connect(self.save_calibration_eeprom)
    
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
    
    calibration_h_folder = QFileDialog.getExistingDirectory(self, 'Select the Folder to which save the calibration.json file')
    with open(os.path.join(str(calibration_h_folder), calibration_json_file_name), 'w') as file:
      json.dump(data, file)
    
    self.ui.set_status('Calibration saved to: ' + str(calibration_h_folder) + calibration_json_file_name + '.\n')
    
    
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
    
    calibration_h_folder = QFileDialog.getExistingDirectory(self, 'Select the Folder to which save the calibration.h file')
    calibration_h_file = open(os.path.join(str(calibration_h_folder), calibration_h_file_name), 'w')
    calibration_h_file.write(calibration_h_text)
    calibration_h_file.close()
    
    self.ui.set_status('Calibration saved to: ' + str(calibration_h_folder) + calibration_h_file_name + ' .\nRecompile and upload the program using the FreeIMU library to your microcontroller.')
  
  def save_calibration_eeprom(self):
    self.ser.write('c'.encode())
    # pack data into a string 
    offsets = struct.pack('<hhhhhh', self.acc_offset[0], self.acc_offset[1], self.acc_offset[2], self.mag_offset[0], self.mag_offset[1], self.mag_offset[2])
    scales  = struct.pack('<ffffff', self.acc_scale[0], self.acc_scale[1], self.acc_scale[2], self.mag_scale[0], self.mag_scale[1], self.mag_scale[2])
    # transmit to microcontroller
    self.ser.write(offsets)
    self.ser.write(scales)
    self.ser.write('\r\n'.encode())
    self.ui.set_status('Calibration saved to microcontroller EEPROM.')
    # debug written values to console
    print('Calibration values read back from EEPROM:')
    self.ser.write('C\r\n'.encode())
    for i in range(4):
      print(self.ser.read_until(b'\n'))
      
  def clear_calibration_eeprom(self):
    self.ser.write('x\r\n'.encode())
    # no feedback expected. we assume success.
    self.ui.set_status('Calibration cleared from microcontroller EEPROM.')
    
  def newData(self, reading):
    
    # only display last reading from burst
    if self.ui.accDisplay.isChecked():
      self.acc_data = np.concatenate((self.acc_data, np.array([[reading[0],reading[1],reading[2]]])), axis=0)

    if self.ui.gyrDisplay.isChecked():
      self.gyr_data = np.concatenate((self.gyr_data, np.array([[reading[3],reading[4],reading[5]]])), axis=0)
    
    if self.ui.magDisplay.isChecked():
     self.mag_data = np.concatenate((self.mag_data, np.array([[reading[6],reading[7],reading[8]]])), axis=0)
    
    self.ui.accXY.plot(x = self.acc_data[:,0], y = self.acc_data[:,1], clear = True, pen='r')
    self.ui.accYZ.plot(x = self.acc_data[:,1], y = self.acc_data[:,2], clear = True, pen='g')
    self.ui.accZX.plot(x = self.acc_data[:,2], y = self.acc_data[:,0], clear = True, pen='b')

    self.ui.gyrXY.plot(x = self.gyr_data[:,0], y = self.gyr_data[:,1], clear = True, pen='r')
    self.ui.gyrYZ.plot(x = self.gyr_data[:,1], y = self.gyr_data[:,2], clear = True, pen='g')
    self.ui.gyrZX.plot(x = self.gyr_data[:,2], y = self.gyr_data[:,0], clear = True, pen='b')
    
    self.ui.magXY.plot(x = self.mag_data[:,0], y = self.mag_data[:,1], clear = True, pen='r')
    self.ui.magYZ.plot(x = self.mag_data[:,1], y = self.mag_data[:,2], clear = True, pen='g')
    self.ui.magZX.plot(x = self.mag_data[:,2], y = self.mag_data[:,0], clear = True, pen='b')
    
    self.acc3D_sp.setData(pos=self.acc_data, color = (1, 1, 1, 1), size=2)
    self.gyr3D_sp.setData(pos=self.gyr_data, color = (1, 1, 1, 1), size=2)
    self.mag3D_sp.setData(pos=self.mag_data, color = (1, 1, 1, 1), size=2)

######################################################################
# Main Program
######################################################################

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FreeIMUCal()
    window.show()
    sys.exit(app.exec_())
