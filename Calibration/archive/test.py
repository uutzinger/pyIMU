
import sys, os
from PyQt4.QtGui import QApplication, QDialog, QMainWindow, QCursor, QFileDialog
from ui_freeimu_cal import Ui_FreeIMUCal
from PyQt4.QtCore import Qt,QObject, pyqtSlot, QThread, QSettings, SIGNAL
import numpy as np
import serial, time
from struct import unpack, pack
from binascii import unhexlify
from subprocess import call
import pyqtgraph.opengl as gl
import cal_lib, numpy

acc_file_name = "acc.txt"

acc_range = 25000

class FreeIMUCal(QMainWindow, Ui_FreeIMUCal):
  def __init__(self):
    QMainWindow.__init__(self)

    # Set up the user interface from Designer.
    self.setupUi(self)
    
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
    self.acc_data = [[], [], []]
    
    # setup graphs
    self.accXY.setXRange(-acc_range, acc_range)
    self.accXY.setYRange(-acc_range, acc_range)
    self.accYZ.setXRange(-acc_range, acc_range)
    self.accYZ.setYRange(-acc_range, acc_range)
    self.accZX.setXRange(-acc_range, acc_range)
    self.accZX.setYRange(-acc_range, acc_range)
    
    self.accXY.setAspectLocked()
    self.accYZ.setAspectLocked()
    self.accZX.setAspectLocked()
    
    
    self.accXY_cal.setXRange(-1.5, 1.5)
    self.accXY_cal.setYRange(-1.5, 1.5)
    self.accYZ_cal.setXRange(-1.5, 1.5)
    self.accYZ_cal.setYRange(-1.5, 1.5)
    self.accZX_cal.setXRange(-1.5, 1.5)
    self.accZX_cal.setYRange(-1.5, 1.5)
    
    self.accXY_cal.setAspectLocked()
    self.accYZ_cal.setAspectLocked()
    self.accZX_cal.setAspectLocked()
        
    self.acc3D.opts['distance'] = 30000
    self.acc3D.show()

    ax = gl.GLAxisItem()
    ax.setSize(x=20000, y=20000, z=20000)
    self.acc3D.addItem(ax)
    
    self.acc3D_sp = gl.GLScatterPlotItem()
    self.acc3D.addItem(self.acc3D_sp)
        
    # axis for the cal 3D graph
    g_a = gl.GLAxisItem()
    g_a.setSize(x=10000, y=10000, z=10000)
    self.acc3D_cal.addItem(g_a)

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
        baudrate=115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS
      )
      
      if self.ser.isOpen():
        print "Arduino serial port opened correctly"
        self.set_status("Connection Successfull. Waiting for Arduino reset...")

        # wait for arduino reset on serial open
        time.sleep(3)
        
        self.ser.write('v') # ask version
        self.set_status("Connected to: " + self.ser.readline()) # TODO: hangs if a wrong serial protocol has been loaded. To be fixed.
        
        self.connectButton.setText("Disconnect")
        self.connectButton.clicked.connect(self.serial_disconnect)
        self.serialPortEdit.setEnabled(False)
        self.serialProtocol.setEnabled(False)
        
        self.samplingToggleButton.setEnabled(True)
        
    except serial.serialutil.SerialException, e:
      self.connectButton.setEnabled(True)
      self.set_status("Impossible to connect: " + str(e))
      
    # restore mouse cursor
    QApplication.restoreOverrideCursor()
    self.connectButton.setEnabled(True)

    
  def serial_disconnect(self):
    print "Disconnecting from " + self.serial_port
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
    print "Starting SerialWorker"
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
    (self.acc_offset, self.acc_scale) = cal_lib.calibrate_from_file(acc_file_name)
    
    # map floats into integers
    self.acc_offset = map(int, self.acc_offset)
    
    # show calibrated tab
    self.tabWidget.setCurrentIndex(1)
    
    #populate acc calibration output on gui
    self.calRes_acc_OSx.setText(str(self.acc_offset[0]))
    self.calRes_acc_OSy.setText(str(self.acc_offset[1]))
    self.calRes_acc_OSz.setText(str(self.acc_offset[2]))
    
    self.calRes_acc_SCx.setText(str(self.acc_scale[0]))
    self.calRes_acc_SCy.setText(str(self.acc_scale[1]))
    self.calRes_acc_SCz.setText(str(self.acc_scale[2]))
    
    
    # compute calibrated data
    self.acc_cal_data = cal_lib.compute_calibrate_data(self.acc_data, self.acc_offset, self.acc_scale)
    
    # populate 2D graphs with calibrated data
    self.accXY_cal.plot(x = self.acc_cal_data[0], y = self.acc_cal_data[1], clear = True, pen='r')
    self.accYZ_cal.plot(x = self.acc_cal_data[1], y = self.acc_cal_data[2], clear = True, pen='g')
    self.accZX_cal.plot(x = self.acc_cal_data[2], y = self.acc_cal_data[0], clear = True, pen='b')
        
    # populate 3D graphs with calibrated data
    acc3D_cal_data = np.array(self.acc_cal_data).transpose()
    
    sp = gl.GLScatterPlotItem(pos=acc3D_cal_data, color = (1, 1, 1, 1), size=2)
    self.acc3D_cal.addItem(sp)
        
    # only display last reading in burst
    self.acc_data[0].append(reading[0])
    self.acc_data[1].append(reading[1])
    self.acc_data[2].append(reading[2])
    
    self.accXY.plot(x = self.acc_data[0], y = self.acc_data[1], clear = True, pen='r')
    self.accYZ.plot(x = self.acc_data[1], y = self.acc_data[2], clear = True, pen='g')
    self.accZX.plot(x = self.acc_data[2], y = self.acc_data[0], clear = True, pen='b')
    
    acc_pos = numpy.array([self.acc_data[0],self.acc_data[1],self.acc_data[2]]).transpose()
    self.acc3D_sp.setData(pos=acc_pos, color = (1, 1, 1, 1), size=2)
    
    magn_pos = numpy.array([self.magn_data[0],self.magn_data[1],self.magn_data[2]]).transpose()
    self.magn3D_sp.setData(pos=magn_pos, color = (1, 1, 1, 1), size=2)


class SerialWorker(QThread):
  def __init__(self, parent = None, ser = None):
    QThread.__init__(self, parent)
    self.exiting = False
    self.ser = ser
    
  def run(self):
    print "sampling start.."
    self.acc_file = open(acc_file_name, 'w')
    self.magn_file = open(magn_file_name, 'w')
    count = 100
    in_values = 9
    reading = [0.0 for i in range(in_values)]
    while not self.exiting:
      self.ser.write('b')
      self.ser.write(chr(count))
      for j in range(count):
        for i in range(in_values):
          reading[i] = unpack('h', self.ser.read(2))[0]
        self.ser.read(2) # consumes remaining '\r\n'
        # prepare readings to store on file
        acc_readings_line = "%d %d %d\r\n" % (reading[0], reading[1], reading[2])
        self.acc_file.write(acc_readings_line)
      # every count times we pass some data to the GUI
      self.emit(SIGNAL("new_data(PyQt_PyObject)"), reading)
      print ".",
    # closing acc and magn files
    self.acc_file.close()
    self.magn_file.close()
    return 
  
  def __del__(self):
    self.exiting = True
    self.wait()
    print "SerialWorker exits.."


app = QApplication(sys.argv)
window = FreeIMUCal()

window.show()
sys.exit(app.exec_())