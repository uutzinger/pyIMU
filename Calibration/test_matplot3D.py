import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class Data3DPlotApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.x_data = []
        self.y_data = []
        self.z_data = []

        self.init_ui()

        # QTimer for real-time updating
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.start(10)  # Update the plot every 500 milliseconds

    def init_ui(self):
        self.setWindowTitle("3D Data Plot Example")
        self.setGeometry(100, 100, 800, 600)

        # Widgets
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.ax.set_xlabel("X Data")
        self.ax.set_ylabel("Y Data")
        self.ax.set_zlabel("Z Data")

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def plot_data(self):
        self.ax.clear()
        self.ax.scatter(self.x_data, self.y_data, self.z_data, c='b', marker='o')
        self.ax.set_xlabel("X Data")
        self.ax.set_ylabel("Y Data")
        self.ax.set_zlabel("Z Data")
        self.canvas.draw()

    def update_data(self):
        # Generate new random data points
        x = np.random.rand(10)
        y = np.random.rand(10)
        z = np.random.rand(10)

        self.x_data = x
        self.y_data = y
        self.z_data = z

        self.plot_data()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Data3DPlotApp()
    window.show()

    sys.exit(app.exec_())