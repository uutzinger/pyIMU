import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QOpenGLWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QOpenGLVersionProfile, QSurfaceFormat
import OpenGL.GL as gl
 
class OpenGLVersionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the OpenGL widget
        self.opengl_widget = QOpenGLWidget(self)
        format = QSurfaceFormat()
        format.setVersion(3, 3)  # Set the desired OpenGL version (e.g., 3.3)
        format.setProfile(QSurfaceFormat.CoreProfile)
        self.opengl_widget.setFormat(format)
        self.setCentralWidget(self.opengl_widget)

        # Retrieve OpenGL version
        self.opengl_version = self.get_opengl_version()

    def get_opengl_version(self):
        version = gl.glGetString(gl.GL_VENDOR)
        return version.decode("utf-8")

def main():
    app = QApplication(sys.argv)
    window = OpenGLVersionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
