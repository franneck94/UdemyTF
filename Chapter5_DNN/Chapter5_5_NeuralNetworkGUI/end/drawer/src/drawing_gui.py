import os
import sys

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import uic

from .dnn import build_model
from .dnn import nn_predict
from .preprocessing import get_image


FILE_PATH = os.path.abspath(__file__)
PROJECT_DIR = os.path.dirname(os.path.dirname(FILE_PATH))
IMAGE_DIR = os.path.join(PROJECT_DIR, 'ressources', 'imgs')
GUI_DIR = os.path.join(PROJECT_DIR, 'ressources', 'gui')
MODEL_FILE_PATH = os.path.join(PROJECT_DIR, "ressources", "weights", "dnn_mnist.h5")

# Load the UI File
GUI_MODEL = os.path.join(GUI_DIR, 'GUI.ui')
FORM, BASE = uic.loadUiType(GUI_MODEL)


class Point:
    """Point class for shapes."""
    x, y = 0, 0

    def __init__(self, nx=0, ny=0):
        self.x = nx
        self.y = ny


# Single shape class
class Shape:
    location = Point()
    number = 0

    def __init__(self, L, S):
        self.location = L
        self.number = S


# Cotainer Class for all shapes
class Shapes:
    shapes: list = []

    def __init__(self):
        self.shapes = []

    # Returns the number of shapes
    def NumberOfShapes(self):
        return len(self.shapes)

    # Add a shape to the database, recording its position
    def NewShape(self, L, S):
        shape = Shape(L, S)
        self.shapes.append(shape)

    # Returns a shape of the requested data.
    def GetShape(self, Index):
        return self.shapes[Index]


# Class for painting widget
class Painter(QtWidgets.QWidget):
    ParentLink = 0
    MouseLoc = Point(0, 0)
    LastPos = Point(0, 0)

    def __init__(self, parent):
        super().__init__()
        self.ParentLink = parent
        self.MouseLoc = Point(0, 0)
        self.LastPos = Point(0, 0)

    # Mouse down event
    def mousePressEvent(self, event=None):
        self.ParentLink.IsPainting = True  # type: ignore
        self.ParentLink.ShapeNum += 1  # type: ignore
        self.LastPos = Point(0, 0)

    # Mouse Move event
    def mouseMoveEvent(self, event=None):
        if self.ParentLink.IsPainting == True:  # type: ignore
            self.MouseLoc = Point(event.x(), event.y())
            if (self.LastPos.x != self.MouseLoc.x) and (self.LastPos.y != self.MouseLoc.y):
                self.LastPos = Point(event.x(), event.y())
                self.ParentLink.DrawingShapes.NewShape(self.LastPos, self.ParentLink.ShapeNum)  # type: ignore
            self.repaint()

    # Mose Up Event
    def mouseReleaseEvent(self, event=None):
        if self.ParentLink.IsPainting == True:  # type: ignore
            self.ParentLink.IsPainting = False  # type: ignore

    # Paint Event
    def paintEvent(self, event):
        painter = QtGui.QPainter()
        painter.begin(self)
        self.drawLines(event, painter)
        painter.end()

    # Draw the line
    def drawLines(self, event, painter):
        for i in range(self.ParentLink.DrawingShapes.NumberOfShapes() - 1):  # type: ignore
            T = self.ParentLink.DrawingShapes.GetShape(i)  # type: ignore
            T1 = self.ParentLink.DrawingShapes.GetShape(i + 1)  # type: ignore
            if T.number == T1.number:
                pen = QtGui.QPen(QtGui.QColor(0, 0, 0), 7, QtCore.Qt.SolidLine)
                painter.setPen(pen)
                painter.drawLine(T.location.x, T.location.y, T1.location.x, T1.location.y)


# Main UI Class
class CreateUI(BASE, FORM):  # type: ignore
    # type: ignore
    DrawingShapes = Shapes()
    PaintPanel = 0
    IsPainting = False
    ShapeNum = 0

    def __init__(self):
        # Set up main window and widgets
        super().__init__()
        self.setupUi(self)
        self.setObjectName('Rig Helper')
        self.PaintPanel = Painter(self)
        self.PaintPanel.close()
        self.DrawingFrame.insertWidget(0, self.PaintPanel)
        self.DrawingFrame.setCurrentWidget(self.PaintPanel)
        # Set up Label for on hold picture
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(460, 70, 280, 280))
        default_image_path = os.path.join(IMAGE_DIR, str(-1) + ".png")
        self.pixmap = QtGui.QPixmap(default_image_path)
        self.label.setPixmap(self.pixmap)
        self.Clear_Button.clicked.connect(self.ClearSlate)
        self.Predict_Button.clicked.connect(self.PredictNumber)
        # NN Model
        num_features = 784
        num_targets = 10
        self.model = build_model(num_features, num_targets)
        if os.path.exists(MODEL_FILE_PATH):
            self.model.load_weights(MODEL_FILE_PATH)
        else:
            raise FileNotFoundError("Weights file not found!")

    # Reset Button
    def ClearSlate(self):
        self.DrawingShapes = Shapes()
        self.PaintPanel.repaint()  # type: ignore
        default_image_path = os.path.join(IMAGE_DIR, str(-1) + ".png")
        self.pixmap = QtGui.QPixmap(default_image_path)
        self.label.setPixmap(self.pixmap)

    # Predict Button
    def PredictNumber(self):
        image = get_image(self.DrawingFrame)
        y_pred_class_idx = nn_predict(self.model, image=image)
        image_file_path = os.path.join(IMAGE_DIR, str(y_pred_class_idx) + ".png")
        self.pixmap = QtGui.QPixmap(image_file_path)
        self.label.setPixmap(self.pixmap)


def main_gui() -> int:
    app = QtWidgets.QApplication(sys.argv)
    main_window = CreateUI()
    main_window.show()
    sys.exit(app.exec_())
