#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 17:04:54 2016

@author: Jason
"""

from PyQt5.uic import loadUiType
import pkg_resources
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from DriftModelRT.DriftForager import DriftForager
from DriftModelRT.PreyType import PreyType
from ModelSetResult import ModelSetResult
import os
import csv
import pickle
import sys

def resource_path(relative_path): ## Function necessary for Pyinstaller to find .ui file during compilation
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

ui_path = resource_path("MainUI.ui")

Ui_MainWindow, QMainWindow = loadUiType(ui_path) # Load the classes defined by the UI file, from which the main window inherits

class MainWindow(QMainWindow, Ui_MainWindow):
    
    def __init__(self, app):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.app = app
        self.status("Welcome to BioenergeticHSC (version 1.0).")
        self.currentResult = None
        self.setDefaults()
        # Tell buttons what should happen when they're clicked
        self.btnDriftDensityFile.clicked.connect(lambda: self.chooseFile('drift density'))
        self.btnBatchMethod1File.clicked.connect(lambda: self.chooseFile('batch method 1'))
        self.btnBatchMethod2File.clicked.connect(lambda: self.chooseFile('batch method 2'))
        self.btnRunModel.clicked.connect(lambda: self.runModel(shouldShowPlots=True, shouldConfigureForager=True))
        self.btnRunModelOnBatchMethod1.clicked.connect(self.runBatchMethod1)
        self.btnRunModelOnBatchMethod2.clicked.connect(self.runBatchMethod2)
        self.btnShowDefaultCurves.clicked.connect(self.showDefaultCurves)
        self.hsDepthForVelocityPlot.valueChanged.connect(lambda: self.plotSliceSliderChanged('depth'))
        self.hsVelocityForDepthPlot.valueChanged.connect(lambda: self.plotSliceSliderChanged('velocity'))
        # Tell menu items what should happen when they're clicked
        self.action33_cm_Arctic_Grayling.triggered.connect(lambda: self.loadFishPreset('33 cm Arctic Grayling'))
        self.action18_cm_Dolly_Varden.triggered.connect(lambda: self.loadFishPreset('18 cm Dolly Varden'))
        self.action6_cm_Chinook_Salmon.triggered.connect(lambda: self.loadFishPreset('6 cm Chinook Salmon'))
        self.actionFast_calculation_grid.triggered.connect(lambda: self.loadGridPreset('Fast Calculation Grid'))
        self.actionMedium_calculation_grid.triggered.connect(lambda: self.loadGridPreset('Medium Calculation Grid'))
        self.actionFine_calculation_grid.triggered.connect(lambda: self.loadGridPreset('Fine Calculation Grid'))
        self.actionReset_all_to_defaults.triggered.connect(self.setDefaults)
        self.actionClear_status_log.triggered.connect(self.clearStatusLog)
        self.actionExport_Depth_and_Velocity_Suitability_Spreadsheet.triggered.connect(lambda: self.exportResult('spreadsheet'))
        self.actionExport_Depth_Curve_Plot.triggered.connect(lambda: self.exportResult('depth curve plot'))
        self.actionExport_Velocity_Curve_Plot.triggered.connect(lambda: self.exportResult('velocity curve plot'))
        self.actionExport_Depth_Velocity_Response_Surface_Plot.triggered.connect(lambda: self.exportResult('response surface plot'))
        self.actionExport_Status_Log.triggered.connect(self.exportStatusLog)
        self.actionSave_Model_Settings.triggered.connect(self.saveModelSettings)
        self.actionLoad_Model_Settings.triggered.connect(self.loadModelSettings)
        # Set constraints on text input to prevent invalid input, i.e. mass from 0.8 to 30000 g with 2 decimal places
        self.leFishMass.setValidator(QDoubleValidator(0.1, 30000.0, 2, self.leFishMass))
        self.leFishForkLength.setValidator(QDoubleValidator(5.0, 200.0, 2, self.leFishMass))
        self.leMaxWaterVelocity.setValidator(QIntValidator(5, 500, self.leMaxWaterVelocity))
        self.leMaxDepth.setValidator(QIntValidator(5, 500, self.leMaxDepth))
        self.leWaterTemperature.setValidator(QDoubleValidator(0.0, 40.0, 2, self.leWaterTemperature))
        self.leTurbidity.setValidator(QDoubleValidator(0.0, 999.0, 1, self.leTurbidity))
        self.leFocalDepthSpec.setValidator(QDoubleValidator(0.0, 40.0, 2, self.leFocalDepthSpec))
        self.leModelGridSize.setValidator(QIntValidator(1, 50, self.leModelGridSize))
        self.leIntervalVelocity.setValidator(QIntValidator(1, 30, self.leIntervalVelocity))
        self.leIntervalDepth.setValidator(QIntValidator(1, 30, self.leIntervalDepth))
        self.lePreyDetectionProbability.setValidator(QDoubleValidator(0.01, 1.0, 2, self.lePreyDetectionProbability))
        self.leReactionDistanceMultiplier.setValidator(QDoubleValidator(0.01, 1.0, 2, self.leReactionDistanceMultiplier))
        self.leFocalVelocityScaler.setValidator(QDoubleValidator(0.01, 1.0, 2, self.leFocalVelocityScaler))
        self.leRoughness.setValidator(QDoubleValidator(0, 50, 1, self.leRoughness))
        # Tell the response variable picker to check for changes
        self.cbResponseVariableToPlot.currentIndexChanged.connect(self.showPlots)
        # Set up variable to track if there's an active forager configured
        self.foragerIsConfigured = False
        
    def setDefaults(self):
        """ Gives all the options default values for when the program is loaded without opening a previous file. """
        self.leMaxWaterVelocity.setText("70")
        self.leMaxDepth.setText("80")
        self.leTurbidity.setText("0.0")
        self.leDriftDensityFile.setText(pkg_resources.resource_filename(__name__, 'DriftModelRT/resources/DemoPreyTypes1.csv'))
        self.leBatchMethod1File.setText(pkg_resources.resource_filename(__name__, 'DriftModelRT/resources/DemoBatchListMethod1.csv'))
        self.leBatchMethod2File.setText(pkg_resources.resource_filename(__name__, 'DriftModelRT/resources/DemoBatchListMethod2.csv'))
        self.lePreyDetectionProbability.setText("1.0")
        self.leReactionDistanceMultiplier.setText("1.0")
        self.leFocalVelocityScaler.setText("1.0")
        self.leRoughness.setText("5.0")
        self.cbVelocityProfileMethod.setCurrentIndex(0)
        self.ckbOptimizeDiet.setChecked(True)
        self.loadFishPreset('18 cm Dolly Varden')
        self.loadGridPreset('Fast Calculation Grid')
        
    def loadFishPreset(self, whichFish):
        if whichFish == '33 cm Arctic Grayling':
            self.leFishMass.setText("398.0")
            self.leFishForkLength.setText("33.0")
            self.leWaterTemperature.setText("5.0")
            self.leFocalDepthSpec.setText("5")
            self.cbFocalDepthMethod.setCurrentIndex(1)        
            self.cbSwimmingCostSubmodel.setCurrentIndex(0)        
        elif whichFish == '18 cm Dolly Varden':
            self.leFishMass.setText("46.0")
            self.leFishForkLength.setText("18.0")
            self.leWaterTemperature.setText("13.0")
            self.leFocalDepthSpec.setText("0.5")
            self.cbFocalDepthMethod.setCurrentIndex(1)  
            self.cbSwimmingCostSubmodel.setCurrentIndex(0)        
        elif whichFish == '6 cm Chinook Salmon':
            self.leFishMass.setText("2.5")
            self.leFishForkLength.setText("6.0")
            self.leWaterTemperature.setText("10.0")
            self.leFocalDepthSpec.setText("0.5")
            self.cbFocalDepthMethod.setCurrentIndex(0)    
            self.cbSwimmingCostSubmodel.setCurrentIndex(4)
        else:              
            self.statusError("Tried to load preset for fish {0} that doesn't have a preset defined. Ignoring request.".format(whichFish))    
            
    def loadGridPreset(self, whichPreset):
        if whichPreset == 'Fast Calculation Grid':
            self.leModelGridSize.setText("10")
            self.leIntervalVelocity.setText("10")
            self.leIntervalDepth.setText("10")
        elif whichPreset == 'Medium Calculation Grid':
            self.leModelGridSize.setText("5")
            self.leIntervalVelocity.setText("5")
            self.leIntervalDepth.setText("5")
        elif whichPreset == 'Fine Calculation Grid':
            self.leModelGridSize.setText("2")
            self.leIntervalVelocity.setText("2")
            self.leIntervalDepth.setText("2")
        else:              
            self.statusError("Tried to load preset for calculation grid preset {0} that doesn't exist. Ignoring request.".format(whichPreset))
            
            
    def saveModelSettings(self):
        savedSettings = {       'leFishMass' : self.leFishMass.text(),
                           'leFishForkLength': self.leFishForkLength.text(),
                         'leMaxWaterVelocity': self.leMaxWaterVelocity.text(),
                                'leMaxDepth' : self.leMaxDepth.text(),
                        'leWaterTemperature' : self.leWaterTemperature.text(),
                               'leTurbidity' : self.leTurbidity.text(),
                          'leFocalDepthSpec' : self.leFocalDepthSpec.text(),
                           'leModelGridSize' : self.leModelGridSize.text(),
                        'leIntervalVelocity' : self.leIntervalVelocity.text(),
                           'leIntervalDepth' : self.leIntervalDepth.text(),
                'lePreyDetectionProbability' : self.lePreyDetectionProbability.text(),
              'leReactionDistanceMultiplier' : self.leReactionDistanceMultiplier.text(),
                     'leFocalVelocityScaler' : self.leFocalVelocityScaler.text(),
                                'leRoughness': self.leRoughness.text(),
                   'cbVelocityProfileMethod' : self.cbVelocityProfileMethod.currentIndex(),
                        'cbFocalDepthMethod' : self.cbFocalDepthMethod.currentIndex(),
                    'cbSwimmingCostSubmodel' : self.cbSwimmingCostSubmodel.currentIndex(),
                    'cbTurbulenceAdjustment' : self.cbTurbulenceAdjustment.currentIndex(),
                      'cbAssimilationMethod' : self.cbAssimilationMethod.currentIndex(),
                           'ckbOptimizeDiet' : self.ckbOptimizeDiet.isChecked(),
                        'leDriftDensityFile' : self.leDriftDensityFile.text(),
                        'leBatchMethod1File' : self.leBatchMethod1File.text(),
                        'leBatchMethod2File' : self.leBatchMethod2File.text()
                        }
        outFilePath = QtWidgets.QFileDialog.getSaveFileName(self, "Choose a name and location to save the model settings (.hsc file)", os.path.expanduser("~"), "Habitat suitability curve settings (*.hsc)")[0]
        file = open(outFilePath, 'wb')
        pickle.dump(savedSettings, file)
        file.close()
        self.status("Saved model settings to {0}".format(outFilePath))
        
    def loadModelSettings(self):
        inFilePath = QtWidgets.QFileDialog.getOpenFileName(self,"Choose the .hsc file containing saved settings.",os.path.expanduser("~"),"Habitat suitability curve settings (*.hsc)")[0]
        if inFilePath != "": 
            file = open(inFilePath, 'rb')
            savedSettings = pickle.load(file)
            file.close()
            keys = savedSettings.keys()
            if 'leFishMass' in keys: self.leFishMass.setText(savedSettings['leFishMass'])
            if 'leFishForkLength' in keys: self.leFishForkLength.setText(savedSettings['leFishForkLength'])
            if 'leMaxWaterVelocity' in keys: self.leMaxWaterVelocity.setText(savedSettings['leMaxWaterVelocity'])
            if 'leMaxDepth' in keys: self.leMaxDepth.setText(savedSettings['leMaxDepth'])
            if 'leWaterTemperature' in keys: self.leWaterTemperature.setText(savedSettings['leWaterTemperature'])
            if 'leTurbidity' in keys: self.leTurbidity.setText(savedSettings['leTurbidity'])
            if 'leFocalDepthSpec' in keys: self.leFocalDepthSpec.setText(savedSettings['leFocalDepthSpec'])
            if 'leModelGridSize' in keys: self.leModelGridSize.setText(savedSettings['leModelGridSize'])
            if 'leIntervalVelocity' in keys: self.leIntervalVelocity.setText(savedSettings['leIntervalVelocity'])
            if 'leIntervalDepth' in keys: self.leIntervalDepth.setText(savedSettings['leIntervalDepth'])
            if 'lePreyDetectionProbability' in keys: self.leIntervalDepth.setText(savedSettings['lePreyDetectionProbability'])
            if 'leReactionDistanceMultiplier' in keys: self.leIntervalDepth.setText(savedSettings['leReactionDistanceMultiplier'])
            if 'leFocalVelocityScaler' in keys: self.leIntervalDepth.setText(savedSettings['leFocalVelocityScaler'])
            if 'leRoughness' in keys: self.leRoughness.setText(savedSettings['leRoughness'])
            if 'cbVelocityProfileMethod' in keys: self.cbVelocityProfileMethod.setCurrentIndex(savedSettings['cbVelocityProfileMethod'])
            if 'cbFocalDepthMethod' in keys: self.cbFocalDepthMethod.setCurrentIndex(savedSettings['cbFocalDepthMethod'])
            if 'cbSwimmingCostSubmodel' in keys: self.cbSwimmingCostSubmodel.setCurrentIndex(savedSettings['cbSwimmingCostSubmodel'])
            if 'cbTurbulenceAdjustment' in keys: self.cbTurbulenceAdjustment.setCurrentIndex(savedSettings['cbTurbulenceAdjustment'])
            if 'cbAssimilationMethod' in keys: self.cbAssimilationMethod.setCurrentIndex(savedSettings['cbAssimilationMethod'])
            if 'ckbOptimizeDiet' in keys: self.ckbOptimizeDiet.setChecked(savedSettings['ckbOptimizeDiet'])
            if 'leDriftDensityFile' in keys: self.leDriftDensityFile.setText(savedSettings['leDriftDensityFile'])
            if 'leBatchMethod1File' in keys: self.leBatchMethod1File.setText(savedSettings['leBatchMethod1File'])
            if 'leBatchMethod2File' in keys: self.leBatchMethod2File.setText(savedSettings['leBatchMethod2File'])
            self.status("Loaded model settings from {0}".format(inFilePath))
            
    def chooseFile(self, whichFile):
        if whichFile == 'drift density':
            filePath = QtWidgets.QFileDialog.getOpenFileName(self,"Choose the CSV file containing drift density information.",os.path.expanduser("~"),"CSV Files (*.csv)")[0]
            if filePath != "": self.leDriftDensityFile.setText(filePath)
        elif whichFile == 'batch method 1':
            filePath = QtWidgets.QFileDialog.getOpenFileName(self,"Choose the CSV file containing batch specification for method 1.",os.path.expanduser("~"),"CSV Files (*.csv)")[0]
            if filePath != "": self.leBatchMethod1File.setText(filePath)
        elif whichFile == 'batch method 2':
            filePath = QtWidgets.QFileDialog.getOpenFileName(self,"Choose the CSV file containing batch specification for method 2.",os.path.expanduser("~"), "CSV Files (*.csv)")[0]
            if filePath != "": self.leBatchMethod2File.setText(filePath)
        if filePath != "": self.status("Set {0} file to {1}.".format(whichFile, filePath))
        
    def configureForager(self):
        preyTypes = PreyType.loadPreyTypes(self.leDriftDensityFile.text(), self)
        self.modelGridSize = int(self.leModelGridSize.text())
        self.currentForager = DriftForager(self,
                                preyTypes,
                                float(self.leFishMass.text()),
                                float(self.leFishForkLength.text()),
                                float(self.leWaterTemperature.text()),
                                float(self.leTurbidity.text()),
                                float(self.lePreyDetectionProbability.text()),
                                float(self.leReactionDistanceMultiplier.text()),
                                float(self.leFocalVelocityScaler.text()),
                                float(self.leFocalDepthSpec.text()),
                                self.cbFocalDepthMethod.currentIndex(),
                                self.cbVelocityProfileMethod.currentIndex(),
                                self.cbSwimmingCostSubmodel.currentIndex(),
                                self.cbTurbulenceAdjustment.currentIndex(),
                                self.cbAssimilationMethod.currentIndex(),
                                float(self.leRoughness.text())
                                )
        self.foragerIsConfigured = True
        
    def runModel(self, shouldShowPlots=True, shouldConfigureForager=True):
        self.status("Running model...")
        if shouldConfigureForager: self.configureForager()
        self.depthInterval = int(self.leIntervalDepth.text())
        self.velocityInterval = int(self.leIntervalVelocity.text())
        maxDepth = int(self.leMaxDepth.text())
        maxVelocity = int(self.leMaxWaterVelocity.text())
        depths = np.arange(self.depthInterval,maxDepth+0.0001,self.depthInterval) # numpy.arange excludes the max value given, so we add 0.0001 to include maxDepth, etc.
        velocities = np.arange(self.velocityInterval,maxVelocity+0.0001,self.velocityInterval)
        dg, vg = np.meshgrid(depths, velocities)
        dv = np.array([dg.flatten(), vg.flatten()]).T
        self.status("Calculating NREI for {0} depth/velocity combinations.".format(len(dv)))
        results = []
        self.pbModelRunProgress.setMaximum(len(dv)-1)
        self.pbModelRunProgress.setValue(0)
        maxNetRateOfEnergyIntake = -100000 # placeholder to be replaced by highest NREI
        for i in range(len(dv)):
            self.pbModelRunProgress.setValue(i)
            self.app.processEvents() # Forces the progress bar and status window to update with each iteration rather than waiting until the end of the loop.
            depth, velocity = dv[i]
            if self.ckbOptimizeDiet.isChecked():
                result = self.currentForager.runForagingModelWithDietOptimization(depth, velocity, self.modelGridSize)
            else:
                result = self.currentForager.runForagingModel(depth, velocity, self.modelGridSize)
            maxNetRateOfEnergyIntake = result.netRateOfEnergyIntake if result.netRateOfEnergyIntake > maxNetRateOfEnergyIntake else maxNetRateOfEnergyIntake
            results.append(result)
            self.status("Calculated NREI = {0:.4f} j/s at depth = {1:.2f} cm and velocity = {2:.2f} cm/s.".format(result.netRateOfEnergyIntake,depth,velocity))
        for result in results: result.standardizeSuitability(maxNetRateOfEnergyIntake) # Calculate the standardized suitability for each result after the overall maximum is known
        self.status("Completed NREI calculations for {0} depth/velocity combinations with maximun NREI = {1:.2f} J/s.".format(len(dv),maxNetRateOfEnergyIntake))
        self.pbModelRunProgress.setValue(0)   # Reset progress bar
        if shouldShowPlots:
            self.hsDepthForVelocityPlot.setMaximum(maxDepth/self.depthInterval)
            self.hsVelocityForDepthPlot.setMaximum(maxVelocity/self.velocityInterval)
            self.currentResult = ModelSetResult(self, results, maxNetRateOfEnergyIntake)
            self.showPlots()
            self.swResultsControls.setCurrentIndex(1) # Make the sliders/buttons to control the 'Results' plots visible by switching the stacked widget to the non-blank page
            self.mainTabWidget.setCurrentIndex(1) # Switch user to 'Results' tab
        else:
            self.currentResult = ModelSetResult(self, results, maxNetRateOfEnergyIntake)

    def showPlots(self):
        responseDict = {0 : 'netRateOfEnergyIntake',
                        1 : 'standardizedSuitability',
                        2 : 'grossRateOfEnergyIntake',
                        3 : 'captureManeuverCostRate',
                        4 : 'focalSwimmingCostRate',
                        5 : 'totalEnergyCostRate',
                        6 : 'meanReactionDistance', 
                        7 : 'captureSuccess',
                        8 : 'proportionOfTimeSpentHandling',
                        9 : 'ingestionRate',
                        10 : 'encounterRate',
                        11 : 'meanPreyEnergyValue',
                        12 : 'numPreyTypes',
                        13 : 'proportionAssimilated'}
        if self.currentResult: 
            self.currentResult.setResponse(responseDict[self.cbResponseVariableToPlot.currentIndex()])
            self.currentResult.plotSuitabilityCurve('depth', self.hsVelocityForDepthPlot.value()*self.velocityInterval, self.mplDepthLayout)
            self.currentResult.plotSuitabilityCurve('velocity', self.hsDepthForVelocityPlot.value()*self.depthInterval, self.mplVelocityLayout)
            self.currentResult.plotResponseSurface(self.mplDepthAndVelocityLayout)
            self.currentResult.showDefaultCurves()
            
    def runBatchMethod1(self):
        inFilePath = self.leBatchMethod1File.text()
        outFilePath = QtWidgets.QFileDialog.getSaveFileName(self, "Choose a name and location for the output CSV file", os.path.expanduser("~"), ".csv")[0]
        if outFilePath == '':
            self.status("Canceled batch method 1 process because no output file was selected.")
        else:
            inputs = []
            self.configureForager()
            with open(inFilePath, newline='') as csvFile:
                reader = csv.reader(csvFile)
                next(reader, None)  # skip the header row
                for row in reader:
                    try:
                        label = row[0]
                        depth = float(row[1])
                        velocity = float(row[2])
                        try:                            # use custom length/mass if both are specified in CSV file
                            forkLength = float(row[4])
                            mass = float(row[5])
                        except ValueError:         # otherwise use the ones from the main window
                            mass = self.currentForager.mass
                            forkLength = self.currentForager.forkLength
                        try:
                            roughness = float(row[3])
                        except ValueError:
                            roughness = self.currentForager.roughness
                        try:  # use custom temperature if specified in CSV file
                            temperature = float(row[6])
                        except ValueError:  # otherwise use the temperature from the main window
                            temperature = self.currentForager.waterTemperature
                        try:  # use custom turbidity if specified in CSV file
                            turbidity = float(row[7])
                        except ValueError:  # otherwise use the turbidity from the main window
                            turbidity = self.currentForager.turbidity
                        if row[7] != "":
                            if os.path.isfile(row[8]):
                                customDriftFile = row[8]
                            else:
                                customDriftFile = None
                        else:
                            customDriftFile = None
                        inputs.append((label, depth, velocity, roughness, forkLength, mass, temperature, turbidity, customDriftFile))
                    except ValueError as err:
                        self.statusError("Value encountered in batch method 1 input file that could not be converted to a number! Skipping it. Specific error: {0}".format(err))
            if len(inputs) == 0:
                self.statusError("Batch method 1 input file did not contain any rows.")
            else:
                self.status("Calculating NREI for {0} rows of the batch method 1 input file.".format(len(inputs)))
                maxNetRateOfEnergyIntake = -100000
                results = []
                for row in inputs:
                    label, depth, velocity, roughness, forkLength, mass, temperature, turbidity, customDriftFile = row
                    self.currentForager.roughness = roughness
                    self.currentForager.forkLength = forkLength
                    self.currentForager.mass = mass
                    self.currentForager.waterTemperature = temperature
                    self.currentForager.turbidity = turbidity
                    if customDriftFile is not None:
                        self.currentForager.filterPreyTypes(PreyType.loadPreyTypes(customDriftFile, self))
                    self.currentForager.clear_caches()
                    if self.ckbOptimizeDiet.isChecked():
                        result = self.currentForager.runForagingModelWithDietOptimization(depth, velocity, self.modelGridSize)
                    else:
                        result = self.currentForager.runForagingModel(depth, velocity, self.modelGridSize)
                    result.pointLabel = label
                    result.forkLength = forkLength
                    result.mass = mass
                    result.roughness = roughness
                    result.temperature = temperature
                    result.turbidity = turbidity
                    result.driftFile = customDriftFile if customDriftFile is not None else self.leDriftDensityFile.text()
                    results.append(result)
                    maxNetRateOfEnergyIntake = result.netRateOfEnergyIntake if result.netRateOfEnergyIntake > maxNetRateOfEnergyIntake else maxNetRateOfEnergyIntake
                    self.status("Calculated NREI = {0:.4f} J/s at depth = {1:.2f} cm and velocity = {2:.2f} cm/s for point labeled '{3}'.".format(result.netRateOfEnergyIntake,depth,velocity,label))
                for result in results: result.standardizeSuitability(maxNetRateOfEnergyIntake) # Calculate the standardized suitability for each result after the overall maximum is known
                with open(outFilePath, 'wt') as outFile:
                    writer = csv.writer(outFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(['Label',
                                     'Depth (cm)',
                                     'Velocity (cm/s)',
                                     'Roughness (cm)',
                                     'Fork length (cm)',
                                     'Mass (g)',
                                     'Temperature',
                                     'Turbidity',
                                     'Drift file',
                                     'Net rate of energy intake (J/s)',
                                     'Standardized habitat suitability',
                                     'Gross rate of energy intake (J/s)',
                                     'Energy cost of maneuvering (J/s)',
                                     'Energy cost of focal swimming (J/s)',
                                     'Total energy costs (J/s)',
                                     'Mean reaction distance (cm)',
                                     'Prey capture success proportion',
                                     'Proportion of time handling prey',
                                     'Prey ingestion rate (items/s)',
                                     'Prey encounter rate (items/s)',
                                     'Mean prey energy value (J)',
                                     'Number of prey types in diet',
                                     'Proportion of energy assimilated'])
                    for result in results: writer.writerow([result.pointLabel,
                                                            result.depth,
                                                            result.velocity,
                                                            result.roughness,
                                                            result.forkLength,
                                                            result.mass,
                                                            result.temperature,
                                                            result.turbidity,
                                                            result.driftFile,
                                                            result.netRateOfEnergyIntake,
                                                            result.standardizedSuitability,
                                                            result.grossRateOfEnergyIntake,
                                                            result.captureManeuverCostRate,
                                                            result.focalSwimmingCostRate,
                                                            result.totalEnergyCostRate,
                                                            result.meanReactionDistance,
                                                            result.captureSuccess,
                                                            result.proportionOfTimeSpentHandling,
                                                            result.ingestionRate,
                                                            result.encounterRate,
                                                            result.meanPreyEnergyValue,
                                                            result.numPreyTypes,
                                                            result.proportionAssimilated])
                self.status("Saved batch processing results to {0}.".format(outFilePath))

    def runBatchMethod2(self):
        inFilePath = self.leBatchMethod2File.text()
        outFolderPath = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose a folder for the output CSV files", os.path.expanduser("~"), QtWidgets.QFileDialog.ShowDirsOnly)
        if outFolderPath == '':
            self.status("Canceled batch method 2 process because no output folder was selected.")
        else:
            inputs = []
            self.configureForager()
            with open(inFilePath, newline='') as csvFile:
                reader = csv.reader(csvFile)
                next(reader, None)  # skip the header row
                for row in reader:
                    try:
                        label = row[0]
                        forkLength = float(row[1])
                        mass = float(row[2])
                        try:  # use custom temperature if specified in CSV file
                            temperature = float(row[3])
                        except ValueError:  # otherwise use the temperature from the main window
                            temperature = self.currentForager.waterTemperature
                        try:  # use custom turbidity if specified in CSV file
                            turbidity = float(row[4])
                        except ValueError:  # otherwise use the turbidity from the main window
                            turbidity = self.currentForager.turbidity
                        if row[5] != "":
                            if os.path.isfile(row[5]):
                                customDriftFile = row[5]
                            else:
                                customDriftFile = None
                        else:
                            customDriftFile = None
                        inputs.append((label, forkLength, mass, temperature, turbidity, customDriftFile))
                    except ValueError as err:
                        self.statusError("Value encountered in batch method 2 input file that could not be converted to a number! Skipping it. Specific error: {0}".format(err))
            if len(inputs) == 0:
                self.statusError("Batch method 2 input file did not contain any rows.")
            else:
                self.status("Calculating NREIs for {0} rows of the batch method 2 input file.".format(len(inputs)))
                for row in inputs:
                    label, forkLength, mass, temperature, turbidity, customDriftFile = row
                    self.status("Calculating NREI for row with label {0}.".format(label))
                    self.currentForager.mass = mass
                    self.currentForager.forkLength = forkLength
                    self.currentForager.waterTemperature = temperature
                    self.currentForager.turbidity = turbidity
                    if customDriftFile is not None:
                        self.currentForager.filterPreyTypes(PreyType.loadPreyTypes(customDriftFile, self))
                    self.status("Running model for temperature {0}".format(self.currentForager.waterTemperature))
                    self.currentForager.clear_caches()
                    self.runModel(shouldShowPlots=False, shouldConfigureForager=False)
                    self.currentResult.exportSpreadsheet(os.path.join(outFolderPath, "{0} (length {1:.2f} -- mass {2:.2f} -- temp {3:.2f}).csv".format(label, forkLength, mass, temperature)))
                self.status("Saved batch processing results to {0}.".format(outFolderPath))

    def plotSliceSliderChanged(self, whichCurve):
        """ Updates the depth and velocity curves when the user changes the slider to select a different depth or velocity """
        if whichCurve == 'depth':
            interval = int(self.leIntervalDepth.text())
            slider = self.hsDepthForVelocityPlot
            lineEdit = self.leDepthForVelocityPlot
            otherCurve = 'velocity'
        elif whichCurve == 'velocity':
            interval = int(self.leIntervalVelocity.text())
            slider = self.hsVelocityForDepthPlot
            lineEdit = self.leVelocityForDepthPlot 
            otherCurve = 'depth'
        else:
            quit("Invalid curve requested in plotSliceSliderChanged.")
        newValue = slider.value() * interval
        self.currentResult.updateSuitabilityCurve(otherCurve, newValue)
        lineEdit.setText(str(newValue))
        
    def showDefaultCurves(self):
        """ Sets the two 2-D curve views on the results pane to the default values. """
        if self.currentResult: self.currentResult.showDefaultCurves()
                        
    def exportResult(self, whichResult):
        if self.currentResult is not None:
            if whichResult == 'spreadsheet':
                self.currentResult.exportSpreadsheet()    
            elif whichResult == 'response surface plot':
                self.currentResult.exportPlot('response surface')
            elif whichResult == 'depth curve plot':
                self.currentResult.exportPlot('depth curve')
            elif whichResult == 'velocity curve plot':
                self.currentResult.exportPlot('velocity curve')
        else:
            self.statusError("Cannot export results until some results have been calculated.")
        
    def exportStatusLog(self):
        outFilePath = QtWidgets.QFileDialog.getSaveFileName(self, "Choose a name and location for the txt file to save the status log", os.path.expanduser("~"), "Text (*.txt)")[0]
        if outFilePath != '':
            self.status("Saved status log to {0}.".format(outFilePath))
            outFile = open(outFilePath, "w")
            outFile.write(self.statusText.toPlainText())
            outFile.close()
        
    def status(self, text):
        self.statusText.appendPlainText(text)
        
    def statusError(self, text):
        self.status("ERROR: " + text)
        
    def clearStatusLog(self):
        self.statusText.clear()