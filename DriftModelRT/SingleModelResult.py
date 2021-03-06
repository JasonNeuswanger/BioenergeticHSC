#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class holds the result of a single model run and calculates the final model performance metrics and internal
statistics based on the totals of various quantities over a single unit of searching time.
"""

import numpy as np

class SingleModelResult(object):
    
    def __init__(self, depth, velocity, preyTypes, totalHandlingTime, totalEnergyIntake, totalReactionDistance, totalPreyEncountered, totalPreyIngested, totalCaptureManeuverCost, totalFocalSwimmingCost, proportionAssimilated):
        """ All of the 'total' inputs here refer to the total of the given quantity resulting from 1 unit (second) of searching time. """ 
        self.depth = depth
        self.velocity = velocity
        totalTime = 1 + totalHandlingTime  # total time involved is 1 unit of search time plus corresponding handling time
        self.meanReactionDistance = totalReactionDistance / totalPreyEncountered if totalPreyEncountered > 0 else np.nan
        self.grossRateOfEnergyIntake = totalEnergyIntake / totalTime
        self.captureManeuverCostRate = totalCaptureManeuverCost / totalTime  # energy spent on capture maneuvers per unit overall time, not just per unit time of maneuvering
        self.focalSwimmingCostRate = totalFocalSwimmingCost / totalTime  # energy spent on swimming at the focal point per unit overall time, not just per unit time of focal swimming
        self.proportionOfTimeSpentHandling = totalHandlingTime / totalTime
        self.totalEnergyCostRate = self.captureManeuverCostRate + self.focalSwimmingCostRate
        self.netRateOfEnergyIntake = self.grossRateOfEnergyIntake - self.totalEnergyCostRate
        self.ingestionRate = totalPreyIngested / totalTime
        self.encounterRate = totalPreyEncountered / totalTime
        self.meanPreyEnergyValue = totalEnergyIntake / totalPreyIngested if totalPreyIngested > 0 else np.nan  # gross energy intake per item ingested
        self.captureSuccess = self.ingestionRate / self.encounterRate if self.encounterRate > 0 else np.nan
        self.preyTypes = preyTypes
        self.numPreyTypes = len(preyTypes) 
        self.proportionAssimilated = proportionAssimilated
        for preyType in self.preyTypes: preyType.ingestionRate = preyType.ingestionCount / totalTime
        self.pointLabel = None  # for temporary storage of point label when processing from a batch file
        # The following are placeholders for numbers set externally when processing Daily NEI and risk
        self.hour = None
        self.hourlyRisk = None
        self.actuallyForaged = None
        
    def standardizeSuitability(self, maxNetRateOfEnergyIntake):
        ''' This standardizes habitat suitability to a maximum of 1. It's set in a separate function because the standardized suitability for each model result
            can only be calculated after we've got NREIs for all the model results. IMPORTANTLY, suitability is standardized to a maximum of 1 over the entire
            grid of depth and velocity combinations, NOT for each individual curve. So depending on the depth and velocity selected to be shown in each curve,
            the maximum won't necessarily be 1. This might mess with some applications that require a 2-D curve with a maximum of 1, and they should be advised 
            in the user manual to re-standardize whatever particular row or column of the output spreadsheet they're using for such an application. ''' 
        self.standardizedSuitability = self.netRateOfEnergyIntake / maxNetRateOfEnergyIntake

class EmptySingleModelResult:

    def __init__(self, depth, velocity, preyTypes):
        """ Placeholder for batch 3 results calculated on the input points defined at the edges of the grid, with depth/velocity 0. The fish
            can't actually occupy these, but they have to be included in the grid to define the conditions between the edges and adjacent points.
            Only their positionOnTransect is actually important."""
        self.depth = depth
        self.velocity = velocity
        self.meanReactionDistance = np.nan
        self.grossRateOfEnergyIntake = 0
        self.captureManeuverCostRate = 0
        self.focalSwimmingCostRate = 0
        self.proportionOfTimeSpentHandling = 0
        self.totalEnergyCostRate = 0
        self.netRateOfEnergyIntake = 0
        self.ingestionRate = 0
        self.encounterRate = 0
        self.meanPreyEnergyValue = np.nan
        self.captureSuccess = np.nan
        self.preyTypes = preyTypes
        self.numPreyTypes = len(preyTypes)
        self.proportionAssimilated = np.nan
        for preyType in self.preyTypes: preyType.ingestionRate = 0
        self.pointLabel = None  # for temporary storage of point label when processing from a batch file
        self.standardizedSuitability = 0
        # Again, placeholders for hourly analysis below
        self.hourlyRisk = None
        self.actuallyForaged = None

    def standardizeSuitability(self, dummyNREI):
        """ Exists so results from this class can be treated the same as for normal results in the rest of the code."""
        self.standardizedSuitability = 0
