# type: ignore
import os
import sys
from typing import Any

from PyQt6 import QtCore
from PyQt6 import QtGui
from PyQt6 import QtWidgets
from PyQt6 import uic
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtGui import QPen

from .dnn import build_model
from .dnn import nn_predict
from .preprocessing import get_image


FILE_PATH = os.path.abspath(__file__)
PROJECT_DIR = os.path.dirname(os.path.dirname(FILE_PATH))
IMAGE_DIR = os.path.join(PROJECT_DIR, "ressources", "imgs")
GUI_DIR = os.path.join(PROJECT_DIR, "ressources", "gui")
MODEL_FILE_PATH = os.path.join(
    PROJECT_DIR,
    "ressources",
    "weights",
    "dnn_mnist.h5",
)

# Load the UI File
GUI_MODEL = os.path.join(GUI_DIR, "GUI.ui")
FORM, BASE = uic.loadUiType(GUI_MODEL)


class Point:
    """Point class for shapes."""

    x, y = 0, 0

    def __init__(self, nx: int = 0, ny: int = 0) -> None:
        self.x = nx
        self.y = ny


class Shape:
    location = Point()
    number = 0

    def __init__(self, L: Any, S: Any) -> None:  # noqa: N803
        self.location = L
        self.number = S


class Shapes:
    shapes: list = []  # noqa: RUF012

    def __init__(self) -> None:
        self.shapes = []

    # Returns the number of shapes
    def NumberOfShapes(self):  # noqa: N802
        return len(self.shapes)

    # Add a shape to the database, recording its position
    def NewShape(self, L: Any, S: Any):  # noqa: N802, N803
        shape = Shape(L, S)
        self.shapes.append(shape)

    # Returns a shape of the requested data.
    def GetShape(self, Index) -> Any:  # noqa: N802, N803
        return self.shapes[Index]


class Painter(QtWidgets.QWidget):
    ParentLink = 0
    MouseLoc = Point(0, 0)
    LastPos = Point(0, 0)

    def __init__(self, parent: Any) -> None:
        super().__init__()
        self.ParentLink = parent
        self.MouseLoc = Point(0, 0)
        self.LastPos = Point(0, 0)

    # Mouse down event
    def mousePressEvent(self, event=None):  # noqa: ARG002, N802
        self.ParentLink.IsPainting = True
        self.ParentLink.ShapeNum += 1
        self.LastPos = Point(0, 0)

    # Mouse Move event
    def mouseMoveEvent(self, event=None):  # noqa: N802
        if self.ParentLink.IsPainting is True:
            positions = event.position()
            self.MouseLoc = Point(int(positions.x()), int(positions.y()))
            if (self.LastPos.x != self.MouseLoc.x) and (
                self.LastPos.y != self.MouseLoc.y
            ):
                self.LastPos = Point(positions.x(), positions.y())
                self.ParentLink.DrawingShapes.NewShape(
                    self.LastPos,
                    self.ParentLink.ShapeNum,
                )
            self.repaint()

    # Mose Up Event
    def mouseReleaseEvent(  # noqa: N802
        self,
        event=None,  # noqa: ARG002
    ):
        if self.ParentLink.IsPainting is True:
            self.ParentLink.IsPainting = False

    # Paint Event
    def paintEvent(self, event):  # noqa: N802
        painter = QtGui.QPainter()
        painter.begin(self)
        self.drawLines(event, painter)
        painter.end()

    # Draw the line
    def drawLines(self, event, painter):  # noqa: ARG002, N802
        for i in range(self.ParentLink.DrawingShapes.NumberOfShapes() - 1):
            T = self.ParentLink.DrawingShapes.GetShape(i)  # noqa: N806
            T1 = self.ParentLink.DrawingShapes.GetShape(i + 1)  # noqa: N806
            if T.number == T1.number:
                pen = QPen(QColor(0, 0, 0))
                pen.setWidth(7)
                pen.setStyle(Qt.PenStyle.SolidLine)
                painter.setPen(pen)
                painter.drawLine(
                    int(T.location.x),
                    int(T.location.y),
                    int(T1.location.x),
                    int(T1.location.y),
                )


class CreateUI(BASE, FORM):
    DrawingShapes = Shapes()
    PaintPanel = 0
    IsPainting = False
    ShapeNum = 0

    def __init__(self) -> None:
        super().__init__()
        self.setupUi(self)
        self.setObjectName("Rig Helper")
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
            err_msg = "Weights file not found!"
            raise FileNotFoundError(err_msg)

    # Reset Button
    def ClearSlate(self):  # noqa: N802
        self.DrawingShapes = Shapes()
        self.PaintPanel.repaint()
        default_image_path = os.path.join(IMAGE_DIR, str(-1) + ".png")
        self.pixmap = QtGui.QPixmap(default_image_path)
        self.label.setPixmap(self.pixmap)

    # Predict Button
    def PredictNumber(self):  # noqa: N802
        image = get_image(self.DrawingFrame)
        y_pred_class_idx = nn_predict(self.model, image=image)
        image_file_path = os.path.join(
            IMAGE_DIR,
            str(y_pred_class_idx) + ".png",
        )
        self.pixmap = QtGui.QPixmap(image_file_path)
        self.label.setPixmap(self.pixmap)


def main_gui() -> int:
    app = QtWidgets.QApplication(sys.argv)
    main_window = CreateUI()
    main_window.show()
    sys.exit(app.exec())
