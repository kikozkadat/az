import os

print("DISPLAY:", os.environ.get("DISPLAY"))
print("XDG_SESSION_TYPE:", os.environ.get("XDG_SESSION_TYPE"))
print("Wayland display:", os.environ.get("WAYLAND_DISPLAY"))

from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtCore import Qt
import sys

print("PyQt próbálkozás indul")

app = QApplication(sys.argv)
label = QLabel("GUI teszt ablak – ha látod, működik a Qt")
label.setAlignment(Qt.AlignCenter)
label.resize(400, 200)
label.show()

sys.exit(app.exec_())

