#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class creates a grid of squares approximating 1/2 of a hemispherical cross-section, taking advantage of left-right symmetry.
The grid is unique to each prey type. If we ever need the model to accomodate laterally asymmetrical foraging areas, it will be necessary
to change some options here and in DriftForager.py to accommodate asymmetricl grids, and it will generally double computation times.

Unlike many previous implementations of variants of the Hughes & Dill 1990 drift feeding model, we do not bother with 
the messy geometry of calculating the intersections of rectangular cells with a circular boundary. The real boundary
of foraging is more of a fuzzy drop-off in statistical probability of detection/pursuit rather than a perfectly clean
hemispherical cut-off. Therefore, we are fine approximating the hemispherical cross-section of the foraging volume with
a grid of square cells, using the cells centered within the hemispherical foraging radius and discarding those centered
outside it. The computational efficiency of this implementation of the model makes it practical to use a grid of very
small squares (even 1 cm, given enough runtime) to very closely approximate the hemispherical model.

We create an array of equal-sized rectangles from the bottom (z=0) to just beyond the surface (z=depth) and 
from x=0 (focal point) out to x = reaction distance, then exclude those with centers falling outside the foraging area."""

import functools
import numpy as np

class CalculationGrid(object):

    @staticmethod
    @functools.lru_cache(maxsize=2048)
    def velocityAtDepth(velocityProfileMethod,depth,waterDepth,meanColumnVelocity, roughness):
        """ The uniform option assumes water velocity is the same from surface to bottom. The logarithmic option assumes a logarithmic velocity
            profile as described in Hayes et al (2007) page 173, and based on Gorden et al (2002), which bases it on Einsten and Barbarossa (1952).
            Other searching for Einsten and Barbarossa's formula confirms that the log involved is log10 not ln (https://pubs.usgs.gov/pp/0462b/report.pdf). 
            Hayes et al 2007 describes H as the distance above the bottom, but it only appears to make sense in the formula as the distance below the surface.
            The formula works in any velocity unit (output velocity will be same units as input mean velocity), here using cm/s. 
            """
        if velocityProfileMethod == 0: # logarithmic
            k = 10 * roughness if roughness < waterDepth else 10 * waterDepth # bed roughness height in mm -- make this a configurable setting in cm
            R = 0.01 * waterDepth # hydraulic radius, approximated as depth, as per Hayes et al 2007, and converted from cm to m
            H = 0.01 * depth # depth (measured down from surface), converted from cm to m
            vstar = meanColumnVelocity / (5.75 * np.log10(12.27 * R / k)) # Stream Hydrology: An Introduction for Ecologists eqn 6.50
            return 5.75 * np.log10(30 * H / k) * vstar # Hayes et al 2007 eqn 1
        elif velocityProfileMethod == 1: # uniform water velocity throughout
            return meanColumnVelocity

    @staticmethod
    @functools.lru_cache(maxsize=2048)
    def constructCells(preyType, reactionDistance, focalDepth, waterDepth, meanColumnVelocity, velocityProfileMethod, userGridSize, roughness):
        """ This method builds the grid cells. It really contains everything we want to do in __init__, but it has to be 
            abstracted out of __init__ so we can memoize previously calculated results with lru_cache, which vastly speeds
            up the program's calculation of the full model."""
        maxGridSize = min(reactionDistance/5,waterDepth/5) # Set grid size to 1/5 the reaction distance or depth (whichever is finer) if the user specified too coarse a grid
        gridSize = userGridSize if userGridSize < maxGridSize else maxGridSize # Otherwise calculations can fail for lack of grid cells
        xVertices = np.arange(0,reactionDistance+gridSize,gridSize)
        zVertices = np.arange(0,waterDepth+gridSize,gridSize)  
        xCenters = np.array([(xVertices[i] + xVertices[i+1])/2 for i in range(len(xVertices)-1)])
        zCenters = np.array([(zVertices[i] + zVertices[i+1])/2 for i in range(len(zVertices)-1)])
        rectArea = (xVertices[1] - xVertices[0]) * (zVertices[1] - zVertices[0])
        xg, zg = np.meshgrid(xCenters,zCenters)
        xz=np.array([xg.flatten(),zg.flatten()]).T
        focalPoint = (0, waterDepth - focalDepth) # focal z coordinate is the distance above the bottom
        xz_ingrid = [row for row in xz if 0 <= row[1] <= waterDepth and ((row[0] - focalPoint[0])**2 + (row[1] - focalPoint[1])**2)**0.5 <= reactionDistance]
        cells = []
        for x,z in xz_ingrid:
            distance = ((x - focalPoint[0])**2 + (z - focalPoint[1])**2)**0.5
            velocity = CalculationGrid.velocityAtDepth(velocityProfileMethod, waterDepth - z, waterDepth, meanColumnVelocity, roughness)
            cells.append(GridCell(distance,velocity,rectArea))
        return cells
            
    def __init__(self, preyType, reactionDistance, focalDepth, waterDepth, meanColumnVelocity, velocityProfileMethod, userGridSize, roughness):
        self.cells = CalculationGrid.constructCells(preyType, reactionDistance, focalDepth, waterDepth, meanColumnVelocity, velocityProfileMethod, userGridSize, roughness)
        
class GridCell(object):
    """ For now, this class is really just a glorified data type to hold information about each cell. """
    
    def __init__(self, distance, velocity, area):
        self.distance = distance
        self.velocity = velocity
        self.area = area
        self.encounterRate = None  # used later during calculations
        self.captureSuccess = None # used later during calculations
        