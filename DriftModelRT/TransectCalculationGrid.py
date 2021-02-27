#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" This class recreates some of functionality of CalculationGrid, but for values on a transect. """

import functools
import numpy as np

from DriftModelRT.CalculationGrid import GridCell
from DriftModelRT.CalculationGrid import CalculationGrid

class TransectCalculationGrid(object):

    @functools.lru_cache(maxsize=2048)
    def constructCells(self, focalPositionOnTransect, reactionDistance, focalDepth, velocityProfileMethod, userGridSize):
        """ This method builds the grid cells. It really contains everything we want to do in __init__, but it has to be
            abstracted out of __init__ so we can memoize previously calculated results with lru_cache, which vastly speeds
            up the program's calculation of the full model.

            Note that unlike with regular CalculationGrid, this one isn't doing calculations on a half-grid and then doubling.
            It calculates both sides, since they may have different depth/velocity/roughness. An x grid size will in effect
            be doubled here, in comparison to the same size in the symmetric case.
            """
        xGridSize = userGridSize if userGridSize < reactionDistance / 5 else reactionDistance / 5  # Make sure calculations don't fail from too few grid cells
        xVertices = np.arange(-(reactionDistance + xGridSize), reactionDistance + xGridSize, xGridSize)
        xCenters = [(xVertices[i] + xVertices[i + 1]) / 2 for i in range(len(xVertices) - 1)]
        cells = []
        for i, x in enumerate(xCenters):
            xPositionOnTransect = focalPositionOnTransect + x
            depthAtX = float(self.transectInterpolations['depth'](xPositionOnTransect))
            if depthAtX <= 0:
                continue  # skip the rest of cell construction for this point if we're on a transect edge with 0 depth, or on land
            meanColumnVelocityAtX = float(self.transectInterpolations['velocity'](xPositionOnTransect))
            if meanColumnVelocityAtX <= 0:
                continue  # likewise skip this x point if velocity is 0, or less due to any input glitches
            roughnessAtX = float(self.transectInterpolations['roughness'](xPositionOnTransect))
            focalPoint = (0, depthAtX - focalDepth)
            zGridSize = userGridSize if userGridSize < depthAtX / 5 else depthAtX / 5  # Make sure calculations don't fail from too few grid cells
            zVertices = np.arange(0, depthAtX + zGridSize, zGridSize)
            zCenters = [(zVertices[i] + zVertices[i + 1]) / 2 for i in range(len(zVertices) - 1)]
            width = xVertices[i + 1] - xVertices[i]
            for j, z in enumerate(zCenters):
                height = zVertices[j + 1] - zVertices[j]
                rectArea = height * width
                distance = ((x - focalPoint[0]) ** 2 + (z - focalPoint[1]) ** 2) ** 0.5
                velocity = CalculationGrid.velocityAtDepth(velocityProfileMethod, depthAtX - z, depthAtX, meanColumnVelocityAtX, roughnessAtX)
                if 0 < z < depthAtX and ((x - focalPoint[0]) ** 2 + (z - focalPoint[1]) ** 2) ** 0.5 <= reactionDistance:
                    cells.append(GridCell(distance, velocity, rectArea))
        return cells

    def __init__(self, transectInterpolations, focalPositionOnTransect, reactionDistance, focalDepth, velocityProfileMethod, userGridSize):
        """The default symmetryFactor of 2 allows for performing the calculations on half the grid (i.e., to the fish's right) and
            then doubling the results when using a symmetric grid. Grids used by batch method 3, with fish foraging along an asymmetrical
            transect, have a symmetryFactor of 1."""
        self.transectInterpolations = transectInterpolations
        self.cells = self.constructCells(focalPositionOnTransect, reactionDistance, focalDepth, velocityProfileMethod, userGridSize)
        self.symmetryFactor = 1