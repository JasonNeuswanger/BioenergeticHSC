# -*- coding: utf-8 -*-

import numpy as np
import functools
from copy import deepcopy
from DriftModelRT.SingleModelResult import SingleModelResult
from DriftModelRT.CalculationGrid import CalculationGrid

# Defining parameters for the Brett & Glass 1973 swimming model as global variables so there isn't computationally expensive overhead from recreating these nparrays thousands of times 
amr_params = np.array([(439.744, 0.835, -0.5782, 0.1335, -0.0096), (450.73, 1.1548, -0.9083, 0.1983, -0.0136), (461.716, 1.4747, -1.2384, 0.263, -0.0176), (472.702, 1.7945, -1.5684, 0.3278, -0.0216), (483.688, 2.1143, -1.8985, 0.3925, -0.0255), (494.674, 2.4341, -2.2286, 0.4573, -0.0295), (531.434, 2.555, -1.8418, 0.2961, -0.0199), (568.194, 2.6759, -1.455, 0.1349, -0.0102), (604.953, 2.7968, -1.0683, -0.0263, -0.0006), (641.713, 2.9177, -0.6815, -0.1874, 0.009), (678.473, 3.0387, -0.2948, -0.3486, 0.0187), (741.436, 7.4827, -6.8461, 0.7548, -0.0356), (804.399, 11.9268, -13.3975, 1.8581, -0.0899), (867.362, 16.3708, -19.9489, 2.9615, -0.1442), (930.325, 20.8149, -26.5003, 4.0649, -0.1985), (993.288, 25.2589, -33.0516, 5.1682, -0.2528), (977.569, 20.0484, -27.7516, 4.2329, -0.2033), (961.851, 14.8378, -22.4515, 3.2974, -0.1538), (946.133, 9.6273, -17.1515, 2.3621, -0.1044), (930.415, 4.4168, -11.8514, 1.4267, -0.0549), (914.697, -0.7937, -6.5514, 0.4913, -0.0054), (906.587, -0.927, -6.8678, 0.6507, -0.0175), (898.478, -1.0602, -7.1842, 0.8101, -0.0295), (890.368, -1.1935, -7.5006, 0.9696, -0.0416), (882.259, -1.3268, -7.8171, 1.129, -0.0536), (874.149, -1.46, -8.1335, 1.2884, -0.0657)], dtype=np.float64)
smr_params = np.array([[45.5499, -1.6385, -0.3505, 0.0251, -0.0005], [49.6992, -3.7997, 0.2781, -0.0489, 0.0025], [53.8485, -5.9608, 0.9068, -0.1228, 0.0055], [57.9978, -8.1219, 1.5355, -0.1968, 0.0084], [62.1471, -10.283, 2.1641, -0.2708, 0.0114], [66.2965, -12.4441, 2.7928, -0.3448, 0.0144], [71.9988, -12.783, 2.6376, -0.321, 0.0135], [77.7011, -13.1218, 2.4824, -0.2972, 0.0126], [83.4034, -13.4606, 2.3272, -0.2735, 0.0117], [89.1057, -13.7995, 2.172, -0.2497, 0.0108], [94.808, -14.1383, 2.0168, -0.2259, 0.0099], [103.716, -13.7171, 1.3149, -0.1193, 0.0051], [112.623, -13.2959, 0.613, -0.0127, 0.0002], [121.531, -12.8747, -0.0889, 0.0938, -0.0046], [130.439, -12.4535, -0.7908, 0.2004, -0.0094], [139.346, -12.0323, -1.4927, 0.307, -0.0142], [151.881, -12.9877, -1.682, 0.3381, -0.0156], [164.415, -13.9431, -1.8713, 0.3692, -0.017], [176.95, -14.8985, -2.0607, 0.4003, -0.0184], [189.484, -15.8539, -2.25, 0.4314, -0.0198], [202.019, -16.8093, -2.4393, 0.4625, -0.0212], [218.934, -19.3705, -1.6324, 0.2911, -0.0107], [235.848, -21.9317, -0.8255, 0.1196, -0.0002], [252.763, -24.4929, -0.0186, -0.0518, 0.0103], [269.678, -27.0541, 0.7884, -0.2233, 0.0207], [286.592, -29.6153, 1.5953, -0.3947, 0.0312]],dtype=np.float64)
u_ms_params = np.array([[16.0807, 9.7663, -0.9914, 0.1493, 0.0005], [16.8226, 11.0578, -1.8682, 0.3386, -0.0115], [17.5645, 12.3492, -2.7451, 0.5279, -0.0235], [18.3064, 13.6407, -3.622, 0.7173, -0.0355], [19.0483, 14.9322, -4.4989, 0.9066, -0.0475], [19.7902, 16.2237, -5.3757, 1.0959, -0.0596], [21.7221, 14.583, -4.4013, 0.9199, -0.0487], [23.6541, 12.9423, -3.427, 0.7439, -0.0378], [25.5861, 11.3016, -2.4526, 0.568, -0.027], [27.5181, 9.6609, -1.4782, 0.392, -0.0161], [29.4501, 8.0202, -0.5038, 0.216, -0.0053], [31.2651, 7.6782, -0.109, 0.1486, -0.0008], [33.0801, 7.3363, 0.2857, 0.0813, 0.0036], [34.8951, 6.9943, 0.6804, 0.014, 0.008], [36.7101, 6.6524, 1.0751, -0.0534, 0.0125], [38.5251, 6.3104, 1.4699, -0.1207, 0.0169], [38.0136, 6.759, 1.1385, -0.0551, 0.0127], [37.5021, 7.2076, 0.8072, 0.0106, 0.0085], [36.9906, 7.6561, 0.4759, 0.0763, 0.0043], [36.4791, 8.1047, 0.1445, 0.1419, 0.0001], [35.9676, 8.5533, -0.1868, 0.2076, -0.004], [35.2118, 9.1098, -0.4654, 0.251, -0.0064], [34.4561, 9.6662, -0.7441, 0.2945, -0.0088], [33.7004, 10.2227, -1.0227, 0.3379, -0.0112], [32.9446, 10.7792, -1.3014, 0.3814, -0.0135], [32.1889, 11.3357, -1.58, 0.4248, -0.0159]], dtype=np.float64)


class DriftForager(object):
    
    def __init__(self, ui, preyTypes, mass, forkLength, waterTemperature, turbidity, reactionDistanceMultiplier, preyDetectionProbability, velocityRefugeMultiplier, focalDepthSpec, focalDepthMethod, velocityProfileMethod, swimmingCostSubmodel,turbulenceAdjustment,assimilationMethod):
        self.ui = ui                               # the main program user interface; should be minimally referenced here except to send status messages
        self.mass = mass                           # mass in grams
        self.forkLength = forkLength               # fork length in cm
        self.filterPreyTypes(preyTypes)            # array of PreyType objects, filtered based on mouth gape / gill raker limitations
        self.waterTemperature = waterTemperature   # water temperature in degrees C
        self.turbidity = turbidity                 # turbidity in NTUs
        self.focalDepthSpec = focalDepthSpec       # the number the user specified for the focal depth
        self.focalDepthMethod = focalDepthMethod   # the method by which the user specified the focal depth (proportion of total depth, or distance above bottom)
        self.maximumSustainableSwimmingSpeed = 36.23 * self.forkLength ** 0.19  # in cm/s, from Hughes & Dill 1990
        self.preyDetectionProbability = preyDetectionProbability  # allows reduction in detection probability as compared to lab experiments, values 0.01 to 1.0, default 1.0 (no effect)
        self.reactionDistanceMultiplier = reactionDistanceMultiplier  # allows reduction in reaction distance as compared to lab experiments, values 0.01 to 1.0, default 1.0 (no effect)
        self.velocityRefugeMultiplier = velocityRefugeMultiplier  # Reduces velocity in total swimming cost calculations, values 0.01 to 1, default to 1 (no effect)
        self.velocityProfileMethod = velocityProfileMethod  # logarithmic or uniform
        self.swimmingCostSubmodel = swimmingCostSubmodel 
        self.turbulenceAdjustment = turbulenceAdjustment
        self.assimilationMethod = assimilationMethod
        self.optimalVelocity = 17.6 * self.mass ** 0.05  # optimal swimming velocity from Stewart et al 1983 via Rosenfeld and Taylor 2009
        self.status("Initialized the DriftForager object.")
        
    def filterPreyTypes(self, preyTypes):
        """ This filters prey types to sizes appropriate to the current fish given its mouth gape and gill raker limitations. It's based on
            equations from Wankowski (1979) as adapted by Hayes et al (2000) and used by Hayes et al (2016) with some adjustments for the prey
            length:diameter ratio of 4.3, which I don't quite understand -- might be good to ask John exactly what that means. They appear to
            use meters for fork length, but the formulas only make sense if fork length is expressed in centimters as we do here. 
            Following Hughes et al 2003 and Hayes et al 2000, prey classes are excluded altogether if they do not fit within anatomical constraints.
            And if a prey class is partially within and partially outside the constraints, its drift density is adjusted to the proportion that
            falls within the constraints, and its size, energy, etc, are adjusted to reflect that proportion.
            """
        minPreyLength = 0.115 * self.forkLength        # min prey length in mm, based on gill raker size 
        maxPreyLength = 1.05 * self.forkLength * 4.3   # max prey length in mm, based on mouth gape
        self.preyTypes = []
        numPreyTypesTrimmed = 0
        for preyType in preyTypes:
            if preyType.minLength > minPreyLength and preyType.maxLength < maxPreyLength: # if the prey type fits totally within the size constraints
                self.preyTypes.append(preyType)
            elif preyType.minLength < minPreyLength and preyType.maxLength > minPreyLength: # if the prey type overlaps the minimum size constraint
                numPreyTypesTrimmed += 1
                preyType.trimToSize(minPreyLength, preyType.maxLength)
                self.preyTypes.append(preyType)
            elif preyType.maxLength > maxPreyLength and preyType.minLength < maxPreyLength: # if the prey type overlaps the maximum size constraint
                numPreyTypesTrimmed += 1
                preyType.trimToSize(preyType.minLength, maxPreyLength)
                self.preyTypes.append(preyType)
        numPreyTypesExcluded = len(preyTypes) - len(self.preyTypes)
        self.preyTypes.sort(key=lambda x: x.energyContent)
        if numPreyTypesExcluded > 0 or numPreyTypesTrimmed > 0: 
            self.status("Excluded {0} and trimmed {3} prey types on due to gill raker (>{1:.2f} mm) or mouth gape (<{2:.2f} mm) constraints.".format(numPreyTypesExcluded,minPreyLength,maxPreyLength,numPreyTypesTrimmed))
      
    # Note to future coders: functools.lru_cache() is a Python 'decorator' for use in 'memoizing' (not 'memorizing') results. It basically saves
    # the result of a function call so it doesn't have to be recalculated when called again with the same parameters. It vastly improves speed when 
    # used in the right places. Google 'memoization' for details.
    @functools.lru_cache(maxsize=2048) 
    def focalDepth(self, waterDepth):
        """ Returns the fish's actual depth (distance below the surface in cm) based on the depth specified by the user and method used to specify it. """
        if self.focalDepthMethod == 0: # depth specified as a proportion of water column depth, with surface = 0, bottom = 1
            if self.focalDepthSpec <= 1.0:
                return waterDepth * self.focalDepthSpec
            else:
                self.status("ERROR: Specified focal depth of {0:.2f} should be between 0 and 1 for method 'proportion of water column depth.' Assuming 0.5 by default.".format(self.focalDepthSpec))
                return waterDepth * 0.5
        elif self.focalDepthMethod == 1: # depth specified as a fixed distance above the bottom
            if waterDepth > self.focalDepthSpec:
                return waterDepth - self.focalDepthSpec
            else:
                return waterDepth * 0.1 # default to 0.1 * water depth, i.e. just below the surface, if the specified distance above the bottom would put the fish in the air
        else:
            exit("Focal depth method specified incorrectly.")
            
    @functools.lru_cache(maxsize=2048)
    def reactionDistance(self, preyType):
        """ Reaction distance in cm based on prey length (mm) and fish's fork length (cm). The baseReactionDistance equation
            comes from Hughes & Dill (1990). The turbidity adjustment is from Hayes et al 2016, based on a curve given by Gregory 
            and Northcote 1993. The GN93 curve was for reaction distance of juvenile Chinook salmon. Hayes et al 2016 divided it
            by its maximum value (36) to turn it from a literal reaction distance to a turbidity multiplier. I added a point that 
            if turbidity is extremely low (before about 0.47) the turbidity is set to a value that would make this multiplier equal
            to 1 to a very high precision, i.e. turbidity has no effect. Without this cuttoff, setting turbidity = 0 to just ignore
            turbidity gives an infinite reaction distance."""
        baseReactionDistance = 12 * preyType.length * (1 - np.exp(-0.2 * self.forkLength))  
        turbidity = 0.4703560632 if self.turbidity < 0.4703560632 else self.turbidity 
        turbidityAdjustment = (31.64 - 13.31 * np.log10(turbidity)) / 36
        #self.status("Reaction distance of {0:.2f} cm for prey type of mean length {1:.2f}.".format(baseReactionDistance, preyType.length))
        return baseReactionDistance * turbidityAdjustment  * self.reactionDistanceMultiplier
        
    @functools.lru_cache(maxsize=2048)
    def maximumCaptureDistance(self, preyType, waterVelocity):
        """ Maximum distance (measured in cm in the plane perpendicular to the focal point) at which the fish can capture prey.  
            Source: Hughes & Dill 1990 """
        rd = self.reactionDistance(preyType)
        return np.sqrt(rd**2 - (waterVelocity * rd/self.maximumSustainableSwimmingSpeed)**2)
        
    @functools.lru_cache(maxsize=32768)
    def captureSuccess(self, preyType, waterVelocity, preyDistance):
        """" Both methods are given in Rosenfeld & Taylor 2009, based on data from Hill & Grossman 1993 """
        V = waterVelocity
        d = preyDistance
        RD = self.reactionDistance(preyType)
        FL = self.forkLength # in cm
        T = self.waterTemperature
        u = 1.28 - 0.0588 * V + 0.383 * FL - 0.0918 * (d/RD) - 0.210 * V * (d/RD)
        return np.exp(u) / (1 + np.exp(u))
    
    @functools.lru_cache(maxsize=32768)
    def handlingStats(self, preyType, preyVelocity):
        """ This comes from Hayes et al 2016, who assumed the fish travels a fixed distance (relative to the water) to catch the prey, 
            at a speed equal to the speed the prey is drifting (as opposed to maximum sustainable swimming speed used by Hughes and Dill 1990, etc). 
            The length of the pursuit is assumed to be 2/3 of the reaction distance to that prey type. However, pursuits of prey detected in
            faster water (in the presence of a vertical velocity gradient) may cost more than pursuits of the same distance in slow water
            because the prey speed is higher.

            An addition to what Hayes et al 2016 did here comes from Rosenfeld and Taylor's (2009) use of a slower, optimal velocity for the return
            leg of the maneuver, which generally matches observations. I just assume the return distance equals the pursuit distance (relative to the
            water).
            """
        pursuitDistance = (2/3) * self.reactionDistance(preyType)
        pursuitTime = pursuitDistance / preyVelocity    
        returnDistance = pursuitDistance # assumption, for now
        returnTime = returnDistance / self.optimalVelocity # return time is not factored into "handling time", but it is counted toward maneuver unsteady swimming costs
        unsteadyPursuitVelocity = np.sqrt(3.0 * preyVelocity**2) # effective velocity used for pursuit cost to account for unstady swimming, Hayes et al 2016 eqn 8
        unsteadyReturnVelocity = np.sqrt(3.0 * self.optimalVelocity**2) # effective velocity used for return cost to account for unstady swimming, Hayes et al 2016 eqn 8
        turnCostFactor = 0.978 * np.exp(0.222 * 0.01 * preyVelocity) # factor in the additional cost of turning beyond that of unsteady swimming, Hayes et al 2016 eqn 9, with cm-to-m conversion 0.01
        swimmingCost = (pursuitTime * self.swimmingCost(unsteadyPursuitVelocity) + returnTime * self.swimmingCost(unsteadyReturnVelocity))  * turnCostFactor
        #self.status("Individual maneuver has swimming cost {0:.2f} based on swimming {3:.2f} s at unsteady velocity {1:.2f} for velocity {2:.2f} with turn cost factor {4:.2f}.".format(swimmingCost,unsteadyVelocity,gridCell['velocity'],totalTime,turnCostFactor))
        return (pursuitTime, swimmingCost) # returned tuple contains the "handling time" (s) and energy cost (J) of one maneuver
    
    @functools.lru_cache(maxsize=2048)
    def swimmingCost(self, velocity):
        """ This function calls out to the selected swimming cost model and applies a turbulence scalar if needed. """
        if self.turbulenceAdjustment == 0: # No turbulence adjustment
            turbulence_scalar = 1
        elif self.turbulenceAdjustment == 1: # Rosenfeld et al 2014, equation for no shelter from turbulence, meant for fish from 2.5 to 15 cm long
            turbulence_scalar = (10**(0.45 * velocity / self.forkLength - 0.745)) + 0.82
        elif self.turbulenceAdjustment == 2: # Rosenfeld et al 2014, equation WITH shelter from turbulence, meant for fish from 2.5 to 15 cm long
            turbulence_scalar = 10**(0.050 * (velocity / self.forkLength)**2 - 0.0069)
        if self.swimmingCostSubmodel == 0:
            return self.swimmingCostHayesEtAl(velocity) * turbulence_scalar
        elif self.swimmingCostSubmodel == 1:
            return self.swimmingCostTrudelWelchSteelhead(velocity) * turbulence_scalar
        elif self.swimmingCostSubmodel == 2:
            return self.swimmingCostTrudelWelchSockeye(velocity) * turbulence_scalar
        elif self.swimmingCostSubmodel == 3:
            return self.swimmingCostTrudelWelchCoho(velocity) * turbulence_scalar
        elif self.swimmingCostSubmodel == 4:
            return self.swimmingCostTrudelWelchChinook(velocity) *  turbulence_scalar


    def swimmingCostHayesEtAl(self, velocity):
        """ Based on Hayes et al 2016, which is based mainly on parameters for brown trout from Elliott (1976) and rainbow trout from 
            Rand et al (1993). Note that this equation appears correctly in Hayes et al 2016, but an incorrect version of the same equation
            with a typo appears in some previous papers. It gives swimming cost (J/s) based on velocity (m/s in their paper, converted
            here from cm/s), mass (g), and temperature (C)."""
        a = 4.126 if self.waterTemperature < 7.1 else 8.277
        b1 = 0.734 if self.waterTemperature < 7.1 else 0.731
        b2 = 0.192 if self.waterTemperature < 7.1 else 0.094
        b3 = 2.34
        return a * self.mass**b1 * np.exp(b2*self.waterTemperature) * np.exp(b3*0.01*velocity) * 4.1868 / 86400
    
    def brett_glass_regression_value(self, params): ## Note, the Brett and Glass model has been swapped out for the Trudel and Welch models.
        """ Calculates intermediate quantities for swimming costs using Hughes & Kelly's (1996) tabular reformulation of Brett & Glass's (1972) graphical model 
            of swimming costs. The params variable will hold one of the three sets of parameters defined at the top of the file. One odd feature of this 
            function (added by Jason Neuswanger) that mass is rounded up to 0.8 grams for fish smaller than that. The model was formulated based on sockeye 
            salmon from 2 to 2000 g. Numerical exploration shows that it extrapolates in a sensible manner down to somewhere between 0.4 to 0.8 g, dependeing on
            temperature. However, somewhere in that range, the relationships reverse direction and begin to show a slightly reductin in mass increasing swimming
            costs similarly to adding several grams of mass, instead of decreasing. Eventually it shoots off to infinity. The threshold of 0.8 g appears to play 
            it safe and avoid all this misbehavior regardless of temperature. Rounding smaller fish to 0.8 g gives more realistic results for smaller fish than
            using the model's anomalous extrapolations in that range. Swimming costs for such small fish tend to be very small, anyway, so this difference doesn't
            matter very much. But users studying tiny fish should be warned about it in the manual."""
        t = int(round(self.waterTemperature))
        b1 = params[t,0]
        b2 = params[t,1]
        b3 = params[t,2]
        b4 = params[t,3]
        b5 = params[t,4]
        mass = 0.8 if self.mass < 0.8 else self.mass # see note above for explanation
        return b1 + b2*np.log(mass) + b3*np.log(mass)**2 + b4*np.log(mass)**3 + b5*np.log(mass)**4

    def swimmingCostBrettGlass(self, velocity):
        """ THIS MODEL IS NO LONGER INCLUDED AS AN OPTION. Calculates swimming cost based on parameters from Brett & Glass's (1973) graphical model as digitized into a table by Hughes & Kelly (1996).
            Returns swimming cost per unit time (J/s) at the given velocity (cm/s), which equals the water velocity when holding steady at a focal point. """
        u_ms = self.brett_glass_regression_value(u_ms_params) # maximum sustainable swimming speed
        SMR = self.brett_glass_regression_value(smr_params)   # standard metabolic rate (mgO2 * kg/h)
        AMR = self.brett_glass_regression_value(amr_params)   # active metabolic rate (mgO2 * kg/h), i.e. oxygen consumption at u_ms  
        oq = 14.1 # oxycaloric equivalent in units (j*mgO2) taken as 14.1 from Videler 1993
        mass = 0.8 if self.mass < 0.8 else self.mass # see note on brett_glass_regression_value method above for explanation
        return (1/3600.0) * (mass/1000.0) * oq * np.exp(np.log(SMR) + velocity*((np.log(AMR)-np.log(SMR))/u_ms))

    def swimmingCostTrudelWelchSteelhead(self, velocity):
        """ Calculates swimming cost based on steelhead parameters from regression from Trudel and Welch (2005)"""
        oq = 14.1 # oxycaloric equivalent in units (j*mgO2) taken as 14.1 from Videler 1993
        return (1/3600.0) * oq * np.exp(-1.71 + (0.8*np.log(self.mass))+(0.016*velocity)+(0.046*self.waterTemperature))

    def swimmingCostTrudelWelchSockeye(self, velocity):
        """Calculates swimming cost based on sockeye parameters from regression from Trudel and Welch (2005). Note that swimming costs and SMR are calculated separately"""
        oq = 14.1  # oxycaloric equivalent in units (j*mgO2) taken as 14.1 from Videler 1993
        smr = (1 / 3600.0) * oq * np.exp(-2.94 + (0.87 * np.log(self.mass)) + (0.064 * self.waterTemperature))  ## SMR
        sc = (1 / 3600.0) * oq * np.exp(-6.25 + (0.72 * np.log(self.mass)) + (1.60 * np.log(velocity)))  ## Swimming costs
        return (sc + smr)

    def swimmingCostTrudelWelchCoho(self, velocity):
        """Calculates swimming cost based on sockeye parameters from regression from Trudel and Welch (2005). Then applies empirical ratios to approximate coho swimming costs and SMR"""
        oq = 14.1  # oxycaloric equivalent in units (j*mgO2) taken as 14.1 from Videler 1993
        smr = (1/3600.0) * oq * np.exp(-2.94 + (0.87*np.log(self.mass)) + (0.064 * self.waterTemperature)) ## SMR
        sc = (1/3600.0) * oq * np.exp(-6.25 + (0.72*np.log(self.mass)) + (1.60 * np.log(velocity))) ## Swimming costs
        return (sc + (smr * 1.75))

    def swimmingCostTrudelWelchChinook(self, velocity):
        oq = 14.1  # oxycaloric equivalent in units (j*mgO2) taken as 14.1 from Videler 1993
        smr = (1/3600.0) * oq * np.exp(-2.94 + (0.87*np.log(self.mass)) + (0.064 * self.waterTemperature)) ## SMR
        sc = (1/3600.0) * oq * np.exp(-6.25 + (0.72*np.log(self.mass)) + (1.60 * np.log(velocity))) ## Swimming costs
        return (sc * 0.73) + (smr * 1.15)

    @functools.lru_cache(maxsize=2048)
    def proportionOfEnergyAssimilated(self, energyIntakeRate):
        """ Calculates the proportion of the caloric content of the food source that can actually be assimilated and available for growth or other needs to
             the fish. The input energyIntakeRate should be in J/s, and needs to be converted in this function to something else."""
        assimilationMethod = self.ui.cbAssimilationMethod.currentIndex()
        if assimilationMethod == 0:
            return 0.6                               # Value from Tucker and Rasmussen 1999 and Hewett and Johnson 1992
        elif assimilationMethod == 1:
            return 0.7                               # Value from Elliott 1982 via Hughes et al 2003
        elif assimilationMethod == 2:
            """ This assimilation method calculates the proportion of energy that would be assimilated if the fish were feeding at its current rate for
                an entire day. If it would reach its maximum daily ration in less than a day, the assimilation proportion is calculated based on the maximum
                daily ration, and it is assumed the fish would eventually stop feeding at some point. 
                
                'Consumption is estimated as the proportion of maximum daily ration for a fish at a particular mass and temperature.'
                This is confusing wording. But it means that Cmax is based on ad libitum feeding experiments at the optimum temperature, and
                Cmax * f(T) gives the effective Cmax at any given temperature. And 'p' is the proportion of THAT which the fish actually consumes.
                According to Railsback's "Guidance for Using a Bioenergetics Model to Assess Effects of Stream Temperature," Cmax is temperature
                dependent, which is consistent with what Jordan had. That is, Cmax factors in f(T).
                
                The Fish Bioenergetics 3.0 manual has a nasty habit of describing things as "proportions" and then giving formulas for absolute amounts,
                not proportions. It does the same thing with S. Everything below works in absolute amounts and returns a proportion in the end. """
            # Convert energy intake rate in J/s to consumption in g /g /day based on assumed 5200 cal/g and 4.184 J/calorie = 21757 J/g dry mass,
            # followed by wet mass = 6 * dry mass (Waters 1977, p. 115, Table I), giving 3626 J/g wet mass.
            C = (energyIntakeRate / 3626) *  (60 * 60 * 24) / self.mass # specific consumption rate in g /g /day
            # Parameter values are for Coho and Chinook from Fish Bioenergetics 3.0 manual and Stewart and Ibarra 1992
            CTO = 15    # Water temperature corresponding to 0.98 of the maximun consumption rate
            CTM = 18    # Water temperature at which temperature dependence is still 0.98 of the maximum
            CTL = 24    # Upper water temperature at which temperature dependence is some reduced fraction (CK4) of the maximum rate 
            CQ = 5      # Lower water temperature at which temperature dependence is some reduced fraction (CK1) of the maximum rate 
            CK1 = 0.36  # Unitless fraction, see above
            CK4 = 0.01  # Unitless fraction, see above
            # Equations for consumption temperature dependence based on equation set 3 on page 2-4 of Fish Bioenergetics 3.0 manual
            G1 = (1 / (CTO - CQ)) * np.log((0.98 * (1 - CK1)) / (CK1 * 0.02))
            G2 = (1 / (CTL - CTM)) * np.log((0.98 * (1 - CK4)) / (CK4 * 0.02))
            L1 = np.exp(G1 * (self.waterTemperature - CQ))
            L2 = np.exp(G2 * (CTL - self.waterTemperature))
            Ka = (CK1 * L1) / (1 + CK1 * (L1 - 1))
            Kb = (CK4 * L2) / (1 + CK4 * (L2 - 1))
            f_of_T = Ka * Kb # temperature dependence function for consumption
            # Basic form of consumption function from box on page 2-2 of Fish Bioenergetis 3.0 manual
            CA = 0.303 # intercept of the allometric mass function for a 1 g fish at optimum water temperature
            CB = -0.275 # slope of allometric mass function, i.e. coefficient of mass dependence
            Cmax = CA * (self.mass ** CB) * f_of_T # maximum specific feeding rate (g /g /day)
            if C > Cmax: C = Cmax # IMPORTANT ASSUMPTION -- A fish CAN feed at an instantaneous NREI exceeding Cmax. For assimilation, we assume it doesn't feed above Cmax for the day, i.e. it eventually stops feeding, and we cap C at Cmax. This avoids errors with negative excretion/SDA, etc.
            p = C / Cmax # proportion of maximum consumption
            # Parameters for egestion and excretion based on Equation Set 2 on page 2-7 (clarified from Elliott 1976)
            FA = 0.212  # Intercept of the proportion of consumed energy egested versus water temperature and ration
            FB = -0.222 # Coefficient of water temperature dependence of egestion
            FG = 0.631  # Coefficient for feeding level dependence of egestion
            UA = 0.0314 # These three are the same
            UB = 0.58   # as the above three, but for
            UG = -0.299 # excretion, not egestion
            F = FA * (self.waterTemperature**FB) * np.exp(FG * p) * C       # egestion, g /g /day
            U = UA * (self.waterTemperature**UB) * np.exp(UG * p) * (C - F) # excretion, g /g /day
            # Specific dynamic action (energy spent digesting prey)
            SDA = 0.172         # unitless coefficient for specific dynamic action 
            S = SDA * (C - F)   # S is the assimilated energy lost to specific dynamic action, from page 2-5
            proportionAssimilated = (C - (F + U + S)) / C if C > 0 else 0 # proportion of consumed calories assimilated and available for respiration or growth
            if C > 0 and not 0 < proportionAssimilated < 1:
                self.ui.status("Warning, bad assimilation: with C = {0:8.4f}, F = {1:8.4f}, U = {2:8.4f}, S = {3:8.4f}, and p = {5:4.4f}, fish is assimilating {4:4.4f}".format(C,F,U,S,proportionAssimilated,p))
            return proportionAssimilated

    def clear_caches(self):
        """ This needs to be run anytime a property that affects one of the cached functions, such as mass or temperature, is adjusted. """
        self.reactionDistance.cache_clear()
        self.focalDepth.cache_clear()
        self.maximumCaptureDistance.cache_clear()
        self.swimmingCost.cache_clear()
        self.handlingStats.cache_clear()
        self.proportionOfEnergyAssimilated.cache_clear()
        self.captureSuccess.cache_clear()

    def runForagingModel(self, waterDepth, meanColumnVelocity, gridSize = 10):
        """ Calculates net rate of energy intake and lots of other internal/diagnostic measures. The 'total' variables
            calculated here are totals across all prey types and grid cells per unit (second) of searching time. """
        totalEnergyIntake = 0
        totalCaptureManeuverCost = 0
        totalHandlingTime = 0
        totalPreyIngested = 0
        gridSymmetryFactor = 2 # Doubles effective area of each grid cell to account for the fact that the computation grid only covers half of the symmetric foraging area.
        totalPreyEncountered = 0
        totalReactionDistance = 0
        totalFocalSwimmingCost = self.swimmingCost(CalculationGrid.velocityAtDepth(self.velocityProfileMethod,self.focalDepth(waterDepth),waterDepth,meanColumnVelocity * self.velocityRefugeMultiplier))
        for preyType in self.preyTypes:
            grid = CalculationGrid(preyType, self.reactionDistance(preyType), self.focalDepth(waterDepth), waterDepth, meanColumnVelocity, self.velocityProfileMethod, gridSize)
            preyType.ingestionCount = 0
            for cell in grid.cells:
                cell.captureSuccess = self.preyDetectionProbability * self.captureSuccess(preyType, cell.velocity, cell.distance)
                cell.encounterRate = gridSymmetryFactor * cell.area * cell.velocity * (preyType.driftDensity * 1e-6) # 1e-6 converts prey/m^3 to prey/cm^3
                handlingTime, captureManeuverCost = self.handlingStats(preyType, cell.velocity)
                totalReactionDistance += cell.encounterRate * cell.distance
                totalPreyEncountered += cell.encounterRate
                totalPreyIngested += cell.encounterRate * cell.captureSuccess
                preyType.ingestionCount += cell.encounterRate * cell.captureSuccess
                totalEnergyIntake += cell.encounterRate * cell.captureSuccess * preyType.energyContent
                totalHandlingTime += cell.encounterRate * handlingTime
                totalCaptureManeuverCost += cell.encounterRate * captureManeuverCost
        proportionAssimilated = self.proportionOfEnergyAssimilated(totalEnergyIntake / (1 + totalHandlingTime)) 
        totalAssimilableEnergyIntake = totalEnergyIntake * proportionAssimilated
        return SingleModelResult(waterDepth,meanColumnVelocity, self.preyTypes, totalHandlingTime, totalAssimilableEnergyIntake, totalReactionDistance, totalPreyEncountered, totalPreyIngested, totalCaptureManeuverCost, totalFocalSwimmingCost, proportionAssimilated)

    def runForagingModelWithDietOptimization(self, waterDepth, meanColumnVelocity, gridSize):
        """ Applies the logic of Charnov's optimal diet model, looping through prey types and discarding them if it
            turns out the NREI would be higher without them than with them. Sometimes, a type might not be removed
            at first, but later removals end up suggesting that removing the earlier one would have been beneficial
            as well. For this reason, the 'while' loop runs through until the number of types has stabilized and
            they're all better to leave in the optimal diet than to eliminate."""
        originalPreyTypes = deepcopy(self.preyTypes)
        endPreyTypeCount = None
        startPreyTypeCount = len(self.preyTypes)
        while endPreyTypeCount != startPreyTypeCount:
            startPreyTypeCount = len(self.preyTypes)        
            for preyType in self.preyTypes:
                nreiWithPreyType = self.runForagingModel(waterDepth, meanColumnVelocity, gridSize).netRateOfEnergyIntake
                self.preyTypes.remove(preyType) # temporarily remove the prey type
                nreiWithoutPreyType = self.runForagingModel(waterDepth, meanColumnVelocity, gridSize).netRateOfEnergyIntake
                if nreiWithPreyType > nreiWithoutPreyType: self.preyTypes.insert(0,preyType) # put it back in if it was beneficial   
            endPreyTypeCount = len(self.preyTypes)
        result = self.runForagingModel(waterDepth, meanColumnVelocity, gridSize)
        self.preyTypes = deepcopy(originalPreyTypes) # Restore the forager's prey type options for the next model run
        return result
        
    def status(self, message):
        if self.ui is not None:
            self.ui.status(message)
        else:
            print(message)