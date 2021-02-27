#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Main file for BioenergeticHSC

    ** NOTE, current versions are compiled using Pyinstaller (development version: available at: https://www.pyinstaller.org/downloads.html)
    To compile, set to correct python environment and directory, then run "pyinstaller main.spec" from the command line

    """

if __name__ == '__main__':

    import sys
    import os
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    from MainWindow import MainWindow
    from PyQt5 import QtWidgets, QtCore
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow(app)
    main.show()
    sys.exit(app.exec_())
