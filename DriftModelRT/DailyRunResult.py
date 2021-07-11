#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class holds the results from each of the 24 hours calculated in a daily model run, along with summaries of the
daily run such as daily risk and daily net energy intake. The DailyModelSetResult class in turn can combine a number
of DailyRunResult objects (typically from different depths and velocities) and create graphs based on the DailyRunResult
values in the same way as InstantaneousModelSetResult objects create graphs from SingleModelResult values. This is a bit
confusing because there are also SingleModelResult objects nested inside DailyRunResult. So there are basically 2 layers
to the instantenous model process and 3 in the daily model process:

SingleModelResult --> InstantenousModelSetResult (combining many depths, velocities etc)
SingleModelResult --> DailyRunResult (24 hours of SingleModelResults) --> DailyModelSetResult (combining those over d, v, etc)

The tricky bit is that DailyRunResult has to mimic some aspects of SingleModelResult so that many of the same plotting
functions can be used for both in the ModelSetResult superclass.
"""

class DailyRunResult(object):

    def __init__(self, depth, velocity, hourlyResults, dailyNetEnergyIntake, dailyGrossEnergyIntake, dailyCost, dailyHoursForaging,
                 dailyFocalSwimmingCost, dailyCaptureManeuverCost, dailyRisk, dailyRiskBalancingMetric, dailySpecificConsumption,
                 dailySpecificConsumptionProportional):
        self.depth = depth
        self.velocity = velocity
        self.hourlyResults = hourlyResults
        self.dailyRisk = dailyRisk
        self.dailyRiskOn90DayHorizon = 1 - (1 - self.dailyRisk) ** 90
        self.dailyRiskBalancingMetric = dailyRiskBalancingMetric
        self.dailyNetEnergyIntake = dailyNetEnergyIntake
        self.dailyGrossEnergyIntake = dailyGrossEnergyIntake
        self.dailyCost = dailyCost
        self.dailyFocalSwimmingCost = dailyFocalSwimmingCost
        self.dailyCaptureManeuverCost = dailyCaptureManeuverCost
        self.dailyHoursForaging = dailyHoursForaging
        self.dailySpecificConsumption = dailySpecificConsumption
        self.dailySpecificConsumptionProportional = dailySpecificConsumptionProportional
        self.standardizedSuitability = 0  # Placeholder to be set after all such results are created

    def standardizeSuitability(self, maxDailyNetEnergyIntake, minDailyRiskBalancingMetric, maxDailyRiskBalancingMetric, foragingStrategy):
        if foragingStrategy == 0:  # Standardize by DNEI, higher = better
            if maxDailyNetEnergyIntake > 0:
                self.standardizedSuitability = self.dailyNetEnergyIntake / maxDailyNetEnergyIntake
            else:
                return 0
        elif foragingStrategy == 1:  # Standardize by daily risk-balancing metric
            # This one is tricky, because lower values of the risk/reward ratio are better.
            # Standardized suitability is scaled such that minDailyRiskBalancingMetric maps to 1 and maxDailyRiskBalancingMetric maps to 0
            # Thus the distance from the maximum, divided by the size of the range, defines the standardized suitability.
            sizeOfRiskMetricRange = maxDailyRiskBalancingMetric - minDailyRiskBalancingMetric
            if sizeOfRiskMetricRange == 0:
                return 1  # call it all equally suitable if the fitness metric is the same everywhere, avoids dividing by 0 in that edge case
            else:
                self.standardizedSuitability = (maxDailyRiskBalancingMetric - self.dailyRiskBalancingMetric) / sizeOfRiskMetricRange