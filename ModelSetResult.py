#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class holds the results of a model run (a full 2D table of depth/velocity combination results) 
and handles plotting and exporting of those results.
"""

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # we get a warning message that this is unused, but it actually is necessary for rotating the 3D plot
from matplotlib import cm
from scipy.interpolate import griddata
from PyQt5.QtWidgets import QSpacerItem
from PyQt5 import QtWidgets
from os.path import expanduser
import csv

class ModelSetResult(object):
    
    @staticmethod
    def clearLayout(layout):
        """ Whenever the plots are refreshed, this method is used to delete the previous plot and allow the new plot to replace it rather than stacking into
            the layout below the old one in an ugly grid. """
        while layout.itemAt(0) != None:
            itemToDelete = layout.itemAt(0)
            layout.removeItem(itemToDelete)
            if type(itemToDelete) is not QSpacerItem: 
                itemToDelete.widget().setParent(None)
    
    def __init__(self, ui, results):
        self.ui = ui
        self.results = results
        self.response = 'netRateOfEnergyIntake'
        self.arrangePlotData()
        self.bestVelocityAtMaxDepth = self.velocities[0]  # placeholder
        self.bestNetRateOfEnergyIntakeAtMaxDepth = self.results[0].netRateOfEnergyIntake  # placeholder
        for result in self.results:
            if result.depth == self.depths.max() and result.netRateOfEnergyIntake > self.bestNetRateOfEnergyIntakeAtMaxDepth:
                self.bestVelocityAtMaxDepth = result.velocity
                self.bestNetRateOfEnergyIntakeAtMaxDepth = result.netRateOfEnergyIntake
    
    def arrangePlotData(self):
        self.depths, self.velocities, self.responses = np.array([(result.depth, result.velocity, getattr(result,self.response)) for result in self.results]).T
    
    def setResponse(self, response):
        self.response = response
        self.arrangePlotData()
        
    def responseLabel(self, length):
        """ Returns a 'long' or 'short' label for the current response """
        longLabels = {'netRateOfEnergyIntake'           : "Net rate of energy intake (J/s)",
                      'standardizedSuitability'         : "Standardized habitat suitability",
                      'grossRateOfEnergyIntake'         : "Gross rate of energy intake (J/s)",
                      'captureManeuverCostRate'         : "Energy cost of maneuvering (J/s)",
                      'focalSwimmingCostRate'           : "Energy cost of focal swimming (J/s)",
                      'totalEnergyCostRate'             : "Total energy costs (J/s)",
                      'meanReactionDistance'            : "Mean reaction distance (cm)", 
                      'captureSuccess'                  : "Prey capture success proportion",
                      'proportionOfTimeSpentHandling'   : "Proportion of time handling prey",
                      'ingestionRate'                   : "Prey ingestion rate (items/s)",
                      'encounterRate'                   : "Prey encounter rate (items/s)",
                      'meanPreyEnergyValue'             : "Mean prey energy value (J)",
                      'numPreyTypes'                    : "Number of prey types in diet",
                      'proportionAssimilated'           : "Proportion of energy assimilated"}
        shortLabels = {'netRateOfEnergyIntake'          : "NREI",
                      'standardizedSuitability'         : "Suitability",
                      'grossRateOfEnergyIntake'         : "GREI",
                      'captureManeuverCostRate'         : "Maneuver cost",
                      'focalSwimmingCostRate'           : "Focal cost",
                      'totalEnergyCostRate'             : "Energy costs",
                      'meanReactionDistance'            : "Reaction distance", 
                      'captureSuccess'                  : "Capture success",
                      'proportionOfTimeSpentHandling'   : "Handling time",
                      'ingestionRate'                   : "Ingestion rate",
                      'encounterRate'                   : "Encounter rate",
                      'meanPreyEnergyValue'             : "Prey value",
                      'numPreyTypes'                    : "Prey types",
                      'proportionAssimilated'           : "Assimilation"}    
        if length == 'long':
            return longLabels[self.response]
        elif length == 'short':
            return shortLabels[self.response]
        else:
            return "Invalid label length"
        
    def plotSuitabilityCurve(self, whichCurve, otherCurveValue, layout):
        fig = Figure(facecolor='white')
        ax = fig.gca()
        ax.set_ylabel(self.responseLabel('long'))
        ax.set_ylim([self.responsePlotLimit('min'),self.responsePlotLimit('max')])
        if whichCurve == 'depth':
            curveResults = [result for result in self.results if result.velocity == otherCurveValue]
            if len(curveResults) == 0: exit("Depth curve requested for invalid velocity.")
            x, y = np.array([(result.depth, getattr(result,self.response)) for result in curveResults]).T
            ax.set_title("Depth response for velocity {0} cm/s".format(otherCurveValue))
            ax.set_xlabel('Water depth (cm)')
            ax.canvas = FigureCanvas(fig)
            self.depthLine, = ax.plot(x, y)
            self.depthAx = ax
            self.depthFig = fig
#            This commented-out code could be used as the start of hover-over tooltips at some point in the future.
#            def on_plot_hover(event):
#                if event.xdata is not None: # if actually over the plot
#                    nearestx = min(x, key=lambda var:abs(var-event.xdata)) # x of event in data coords
#                    print(nearestx)
#            fig.canvas.mpl_connect('motion_notify_event', on_plot_hover)  

        elif whichCurve == 'velocity':
            curveResults = [result for result in self.results if result.depth == otherCurveValue]
            if len(curveResults) == 0: exit("Velocity curve requested for invalid depth.")
            x, y = np.array([(result.velocity, getattr(result,self.response)) for result in curveResults]).T
            ax.set_title("Velocity response for depth {0} cm".format(otherCurveValue))
            ax.set_xlabel('Water velocity (cm/s)')
            ax.canvas = FigureCanvas(fig)
            self.velocityLine, = ax.plot(x, y)
            self.velocityAx = ax
            self.velocityFig = fig
        else:
            quit("Invalid curve requested in plotSuitabilityCurve -- should be 'depth' or 'velocity' only.")
        ModelSetResult.clearLayout(layout)
        layout.addWidget(ax.canvas) # need to clear before adding another widget
        fig.tight_layout(pad=2.5)
        ax.canvas.draw()
        
    def responsePlotLimit(self, whichLimit):
        """ Returns minimum and maximum value that shows up on the plot. Sometimes the minimum NREI gets extremely low at high velocity values, so it
            needs to be cut off to preserve detail in the plot in the interesting/realistic velocity ranges."""
        responses = self.responses[np.logical_not(np.isnan(self.responses))]
        responsePadding = 0.03 * (responses.max() - responses.min())
        if whichLimit == 'min':
            if self.response == 'netRateOfEnergyIntake':
                return max(responses.min() - responsePadding,-0.2*responses.max()) # truncate NREI plots so extreme negative values don't squish the plots
            else:
                return responses.min() - responsePadding
                
        elif whichLimit == 'max':
                return responses.max() + responsePadding
        else: 
            return 0
        
    def updateSuitabilityCurve(self, whichCurve, otherCurveValue):
        """ This function updates the depth curve for a given velocity value or vice versa, simply by changing the ydata in the plot 
            to move the line without redrawing the entire figure. """
        if whichCurve == 'depth':
            curveResults = [getattr(result, self.response) for result in self.results if result.velocity == otherCurveValue]
            if len(curveResults) == 0: exit("Depth curve requested for invalid velocity.")
            self.depthLine.set_ydata(np.array(curveResults))
            self.depthAx.set_title("Depth response for velocity {0} cm/s".format(otherCurveValue))
            self.depthAx.canvas.draw()
        elif whichCurve == 'velocity':
            curveResults = [getattr(result, self.response) for result in self.results if result.depth == otherCurveValue]
            if len(curveResults) == 0: exit("Velocity curve requested for invalid depth.")
            self.velocityLine.set_ydata(np.array(curveResults))
            self.velocityAx.set_title("Velocity response for depth {0} cm".format(otherCurveValue))
            self.velocityAx.canvas.draw()
        else:
            quit("Invalid curve requested in updateSuitabilityCurve -- should be 'depth' or 'velocity' only.")        
        
    def showDefaultCurves(self):
        self.ui.hsDepthForVelocityPlot.setValue(self.depths.max() / int(self.ui.leIntervalDepth.text()))
        self.ui.hsVelocityForDepthPlot.setValue(self.bestVelocityAtMaxDepth / int(self.ui.leIntervalVelocity.text()))
            
    def plotResponseSurface(self, layout):
        """ Builds the 3D plot of the full depth/velocity relationship. """
        self.responseSurfaceFig = Figure(facecolor='white')
        canvas = FigureCanvas(self.responseSurfaceFig) # must be created before the add_subplot line
        ax = self.responseSurfaceFig.add_subplot(111, projection='3d')
        ax.canvas = canvas # must be set after the add_subplot line
        ax.mouse_init()
        # 3d plot grid arrangement code adapted from http://stackoverflow.com/questions/29547687/matplotlib-3d-surface-plots-not-showing       
        xi = np.linspace(self.depths.min(), self.depths.max(), 50)
        yi = np.linspace(self.velocities.min(), self.velocities.max(), 50)
        zmethod = 'linear' if np.isnan(self.responses).any() else 'cubic' # cubic interpolation fails with any NaN values, i.e. if the fish aren't eating so rates aren't defined
        zi = griddata((self.depths, self.velocities), self.responses, (xi[None, :], yi[:, None]), method=zmethod)    # create a uniform spaced grid
        xig, yig = np.meshgrid(xi, yi)
        ax.plot_surface(xig, yig, zi, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, vmin=self.responsePlotLimit('min'), vmax=self.responsePlotLimit('max'), antialiased=False)
        ax.set_zlim3d(self.responsePlotLimit('min'),self.responsePlotLimit('max'))
        ax.set_zlabel(self.responseLabel('short'))
        ax.set_xlabel("Depth")
        ax.set_ylabel("Velocity")
        ModelSetResult.clearLayout(layout)
        layout.addWidget(ax.canvas) # need to clear before adding another widget
        ax.canvas.draw()  
        
    def exportPlot(self, whichPlot):
        outFilePath = QtWidgets.QFileDialog.getSaveFileName(self.ui, "Choose a name and location to save the plot (as .png, .pdf, or .jpg)", expanduser("~"), "Image (*.png *.jpg *.pdf)")[0]
        if whichPlot == 'response surface':
            self.responseSurfaceFig.savefig(outFilePath)
        elif whichPlot == 'depth curve':
            self.depthFig.savefig(outFilePath)
        elif whichPlot == 'velocity curve':
            self.velocityFig.savefig(outFilePath)
        else:
            self.ui.statusError("Requested to export non-existent plot.")
        
    def exportSpreadsheet(self, path=None):
        outFilePath = path if path is not None else QtWidgets.QFileDialog.getSaveFileName(self.ui, "Choose a name and location for the output CSV file", expanduser("~"), "Spreadsheet (*.csv)")[0]
        if outFilePath != '':
            depths = np.sort(np.unique(self.depths))
            velocities = np.sort(np.unique(self.velocities))
            dictResponses = {}
            for result in self.results: dictResponses[(result.depth, result.velocity)] = getattr(result,self.response) # store response values in a dict indexed by depth and velocity
            with open(outFilePath, 'wt') as outFile:
                writer = csv.writer(outFile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["velocity (row) by depth (column)"] + depths.tolist())
                for v in velocities:
                    writer.writerow([v] + [dictResponses[(d, v)] for d in depths])
            self.ui.status("Saved responses for the full depth/velocity grid to to {0}.".format(outFilePath))

