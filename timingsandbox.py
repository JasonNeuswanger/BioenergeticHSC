#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file exists only to run NREI requests from the command line in a way that makes it easy to use code profiling
functions, independent of the user interface.
"""

import cProfile, pstats # packages for testing the execution time of model commands
import pkg_resources
from DriftModelRT.DriftForager import DriftForager
from DriftModelRT.PreyType import PreyType

driftDensityFile = pkg_resources.resource_filename(__name__, 'DriftModelRT/resources/DemoPreyTypes.csv')
preyTypes = PreyType.loadPreyTypes(driftDensityFile, None)

forager = DriftForager(None,preyTypes,46,18,13,0,10,1,0,0,0)

def test(nRuns):
    for i in range(nRuns): forager.netRateOfEnergyIntake(50.0, 30.0, 10, True)

cProfile.run('test(1000)','runstats')
p = pstats.Stats('runstats')
p.strip_dirs().sort_stats('cumulative').print_stats()
##p.strip_dirs().sort_stats('cumulative').print_callees()

# was about 2.5 s