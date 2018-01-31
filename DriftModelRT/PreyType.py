# -*- coding: utf-8 -*-

import csv

class PreyType(object):
    
    @staticmethod
    def loadPreyTypes(filePath, ui):
        preyTypes = []
        with open(filePath, newline='') as csvFile:
            reader = csv.reader(csvFile)
            next(reader, None) # skip the header row
            for row in reader:
                try:
                    label = row[0]
                    minLength = float(row[1])
                    maxLength = float(row[2])
                    driftDensity = float(row[3])
                    a = float(row[4]) if row[4] != '' else None
                    b = float(row[5]) if row[5] != '' else None
                    energyDensityCalories = float(row[6]) if row[6] != '' else 5200 # Default energy density is 5200 cal/gm dry wt, from Rosenfeld and Taylor 2009, derived from Cummins & Wuycheck 1971
                    preyTypes.append(PreyType(label, minLength,maxLength,driftDensity,energyDensityCalories,a,b))
                except ValueError as err:
                    message = "Value encountered in drift density file that could not be converted to a number! Skipping a prey type. Error: {0}".format(err)
                    if ui is not None:
                        ui.statusError(message)
                    else:
                        print(message)
        return preyTypes
    
    def __init__(self, label, minLength, maxLength, driftDensity, energyDensityCalories, a = None, b = None):
        self.label = label
        self.minLength = minLength
        self.maxLength = maxLength
        self.a = a if a is not None else 0.0064                        # The values for the W = a L^b regression can be specified in the 4th and 5th columns of the csv,
        self.b = b if b is not None else 2.788                         # but if they're not, the mean value for 'All Insects' from Benke et al 1999 is used.
        self.driftDensity = driftDensity                               # Number of items of prey in this class per m^3 of water.
        self.energyDensityJoules = energyDensityCalories*4.184/1000    # Convert from gram-calories per gram to Joules (4.184J/calorie) per milligram
        self.setDimensions(minLength,maxLength)
        self.ingestionCount = None                                     # Intermediate value for the number below
        self.ingestionRate = None                                      # Rate at which this type is ingested, optionally added by NREI model to keep track
        
    def setDimensions(self, minLength, maxLength):
        self.length = (minLength + maxLength) / 2                      # Mean length in mm of prey in this class.
        self.dryMass = self.a * self.length ** self.b                  # Dry mass (mg), using the most common regression formula
        self.energyContent = self.energyDensityJoules * self.dryMass   # Energy content (J)
        self.energyContent = self.energyContent
        
    def trimToSize(self, minLength,maxLength):
        """ This function adjusts a prey type to account for only part of its length range falling within the anatomical constraints of the forager
            as given by minLength and maxLength. Drift density is reduced to the proportion of the old size range falling within the boundary of the
            new size range, assuming abundance was uniformly distributed within the original size range. """
        sizeRangeWidth = maxLength - minLength
        oldSizeRangeWidth = self.maxLength - self.minLength
        self.driftDensity = self.driftDensity * sizeRangeWidth/oldSizeRangeWidth
        self.setDimensions(minLength,maxLength)
        