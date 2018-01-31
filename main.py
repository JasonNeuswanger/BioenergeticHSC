#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Main
    file
    for BioenergeticHSC

    Cross - platform
    deployment as normal
    executable
    files
    will
    be
    done
    using
    http: // pyqt.sourceforge.net / Docs / pyqtdeploy / command_line.html  # pyqtdeploy

    """

if __name__ == '__main__':

    import sys
    from MainWindow import MainWindow
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow(app)
    main.show()
    sys.exit(app.exec_())
