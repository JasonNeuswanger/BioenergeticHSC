# -*- coding: utf-8 -*-

import numpy as np
import functools
from copy import deepcopy
from DriftModelRT.SingleModelResult import SingleModelResult
from DriftModelRT.SingleModelResult import EmptySingleModelResult
from DriftModelRT.CalculationGrid import CalculationGrid
from DriftModelRT.TransectCalculationGrid import TransectCalculationGrid
from DriftModelRT.PreyType import PreyType
from DriftModelRT.DailyRunResult import DailyRunResult

import datetime
import pytz
from pysolar.solar import get_altitude
from pysolar.radiation import get_radiation_direct
from timezonefinder import TimezoneFinder
import os
import csv

# Note: to install dependencies for daily calculations, run the following:
# conda install -c conda-forge pysolar
# conda install -c conda-forge timezonefinder
# conda install -c conda-forge pytz

def risk_metric_compare(hour_a, hour_b, other_risk=None, other_DNEI=None, risk_0=None):
    """ This function compares the relevant characteristics of two possible hours the fish could forage, hour A and hour B,
        to determine which is better for minimizing the metric (risk_0 + daily_risk) / DNEI.

        The constant risk_0 is included to prevent excessive sensitivity of the combined habitat quality metric to minor variations
        in risk when risk is very low. If we had purely used daily_risk / DNEI, this metric of habitat quality could easily
        double as a very low risk level changes from 'negligible' to 'twice as negligible,' whereas in reality the difference between
        two truly negligible risks should not matter to a fitness metric. The constant risk_0 introduces a sense of scale to the calculation
        to prevent this problem. It can be interpreted as the level of daily risk at which it a fish would need to double its DNEI in
        order to have the same fitness metric as if there were zero risk.

        We use an exact formula derived to compare values that act to minimize the metric, but we never actually know the exact inputs.
        Specifically, other_risk and other_DNEI represent for each input (a and b) the total daily risk and DNEI from all the other
        hours excluding the one being considered as input a or b. However, we do not really know what those will be, because we don't
        know yet which other hours are included or excluded. However, we can use total risk and total DNEI if the fish is foraging
        in every available hour (not excluding the current one) as a rough approximation, which should still put the relative merits
        of each hour in the right order most of the time.

        We're defining "greater than" in this case to mean "having a better contribution to minimizing the overall risk/reward ratio."
        Thus, we will sort descending by this measure to list the best hours first for addition to the foraging schedule.
        This function returns -1 if a < b, 0 if a == b, and 1 if a > b, to fit the definition of a comparison function in Python.

        Note that the last 3 parameters with defaults aren't actually optional, but are specified later using functools.partial"""
    risk_a = hour_a.hourlyRisk
    NREI_a = hour_a.netRateOfEnergyIntake
    risk_b = hour_b.hourlyRisk
    NREI_b = hour_b.netRateOfEnergyIntake
    if risk_a == risk_b and NREI_a == NREI_b:
        return 0
    a_is_better = other_risk * ((other_DNEI + NREI_b) * risk_a - (other_DNEI + NREI_a) * risk_b) > (1 + risk_0) * (NREI_b - NREI_a)  # boolean
    if a_is_better:
        return 1
    else:
        return -1

class DriftForager(object):

    def __init__(self, ui, preyTypes, mass, forkLength, waterTemperature, turbidity, basePreyDetectionProbability, reactionDistanceMultiplier, focalVelocityScaler, focalDepthSpec, focalDepthMethod, velocityProfileMethod, swimmingCostSubmodel, turbulenceAdjustment, assimilationMethod, roughness):
        self.ui = ui  # the main program user interface; should be minimally referenced here except to send status messages
        self.mass = mass  # mass in grams
        self.forkLength = forkLength  # fork length in cm
        if preyTypes is not None:  # pass preyTypes = None to initialize the forager and set prey types later, i.e. for batch runs with different prey type files
            self.filterPreyTypes(preyTypes)  # array of PreyType objects, filtered based on mouth gape / gill raker limitations
        self.waterTemperature = waterTemperature  # water temperature in degrees C
        self.turbidity = turbidity  # turbidity in NTUs
        self.focalDepthSpec = focalDepthSpec  # the number the user specified for the focal depth
        self.focalDepthMethod = focalDepthMethod  # the method by which the user specified the focal depth (proportion of total depth, or distance above bottom)
        self.maximumSustainableSwimmingSpeed = 36.23 * self.forkLength ** 0.19  # in cm/s, from Hughes & Dill 1990
        self.basePreyDetectionProbability = basePreyDetectionProbability  # allows reduction in detection probability as compared to lab experiments, values 0.01 to 1.0, default 1.0 (no effect)
        self.reactionDistanceMultiplier = reactionDistanceMultiplier  # allows reduction in reaction distance as compared to lab experiments, values 0.01 to 1.0, default 1.0 (no effect)
        self.focalVelocityScaler = focalVelocityScaler  # Reduces velocity in focal swimming cost calculations, values 0.01 to 1, default to 1 (no effect)
        self.velocityProfileMethod = velocityProfileMethod  # logarithmic or uniform
        self.swimmingCostSubmodel = swimmingCostSubmodel  # the swimming cost model specified by the user
        self.turbulenceAdjustment = turbulenceAdjustment  # whether focal velocity is adjusted for turbulence (default is on)
        self.assimilationMethod = assimilationMethod  # the energy assimilation method selected
        self.roughness = roughness  # the roughness height in cm
        self.optimalVelocity = 17.6 * self.mass ** 0.05  # optimal swimming velocity from Stewart et al 1983 via Rosenfeld and Taylor 2009
        self.positionOnTransect = None  # placeholder used in batch process 3 when processing data on a transect
        self.hourlyDriftMultiplier = 1  # multiplier optionally used in hourly analyses to reflect time-varying drift; set to 1 for no effect
        self.status("Initialized the DriftForager object.")

    def filterPreyTypes(self, preyTypes):
        """ This filters prey types to sizes appropriate to the current fish given its mouth gape and gill raker limitations. It's based on
            equations from Wankowski (1979) as adapted by Hayes et al (2000) and used by Hayes et al (2016) with some adjustments for the prey
            length:diameter ratio of 4.3, which I don't quite understand -- might be good to ask John exactly what that means. They appear to
            use meters for fork length, but the formulas only make sense if fork length is expressed in centimeters as we do here.
            Following Hughes et al 2003 and Hayes et al 2000, prey classes are excluded altogether if they do not fit within anatomical constraints.
            And if a prey class is partially within and partially outside the constraints, its drift density is adjusted to the proportion that
            falls within the constraints, and its size, energy, etc, are adjusted to reflect that proportion.
            """
        minPreyLength = 0.115 * self.forkLength  # min prey length in mm, based on gill raker size
        maxPreyLength = 1.05 * self.forkLength * 4.3  # max prey length in mm, based on mouth gape
        self.preyTypes = []
        numPreyTypesTrimmed = 0
        for preyType in preyTypes:
            if preyType.minLength > minPreyLength and preyType.maxLength < maxPreyLength:  # if the prey type fits totally within the size constraints
                self.preyTypes.append(preyType)
            elif preyType.minLength < minPreyLength and preyType.maxLength > minPreyLength:  # if the prey type overlaps the minimum size constraint
                numPreyTypesTrimmed += 1
                preyType.trimToSize(minPreyLength, preyType.maxLength)
                self.preyTypes.append(preyType)
            elif preyType.maxLength > maxPreyLength and preyType.minLength < maxPreyLength:  # if the prey type overlaps the maximum size constraint
                numPreyTypesTrimmed += 1
                preyType.trimToSize(preyType.minLength, maxPreyLength)
                self.preyTypes.append(preyType)
        numPreyTypesExcluded = len(preyTypes) - len(self.preyTypes)
        self.preyTypes.sort(key=lambda x: x.energyContent)
        if numPreyTypesExcluded > 0 or numPreyTypesTrimmed > 0:
            self.status("Excluded {0} and trimmed {3} prey types on due to gill raker (>{1:.2f} mm) or mouth gape (<{2:.2f} mm) constraints.".format(numPreyTypesExcluded, minPreyLength, maxPreyLength, numPreyTypesTrimmed))

    # Note to future coders: functools.lru_cache() is a Python 'decorator' for use in 'memoizing' (not 'memorizing') results. It basically saves
    # the result of a function call so it doesn't have to be recalculated when called again with the same parameters. It vastly improves speed when 
    # used in the right places. Google 'memoization' for details.
    @functools.lru_cache(maxsize=2048)
    def focalDepth(self, waterDepth):
        """ Returns the fish's actual depth (distance below the surface in cm) based on the depth specified by the user and method used to specify it. """
        if self.focalDepthMethod == 0:  # depth specified as a proportion of water column depth, with surface = 0, bottom = 1
            if self.focalDepthSpec <= 1.0:
                return waterDepth * self.focalDepthSpec
            else:
                self.status("ERROR: Specified focal depth of {0:.2f} should be between 0 and 1 for method 'proportion of water column depth.' Assuming 0.5 by default.".format(self.focalDepthSpec))
                return waterDepth * 0.5
        elif self.focalDepthMethod == 1:  # depth specified as a fixed distance above the bottom
            if waterDepth > self.focalDepthSpec:
                return waterDepth - self.focalDepthSpec
            else:
                return waterDepth * 0.1  # default to 0.1 * water depth, i.e. just below the surface, if the specified distance above the bottom would put the fish in the air
        else:
            exit("Focal depth method specified incorrectly.")

    @functools.lru_cache(maxsize=2048)
    def reactionDistance(self, preyType):
        """ Reaction distance in cm based on prey length (mm) and fish's fork length (cm). The baseReactionDistance equation
            comes from Hughes & Dill (1990). The turbidity adjustment is from Hayes et al 2016, based on a curve given by Gregory 
            and Northcote (1993). The GN93 curve was for reaction distance of juvenile Chinook salmon. Hayes et al 2016 divided it
            by its maximum value (36) to turn it from a literal reaction distance to a turbidity multiplier. I added a point that 
            if turbidity is extremely low (before about 0.47) the turbidity is set to a value that would make this multiplier equal
            to 1 to a very high precision, i.e. turbidity has no effect. Without this cuttoff, setting turbidity = 0 to just ignore
            turbidity gives an infinite reaction distance."""
        baseReactionDistance = 12 * preyType.length * (1 - np.exp(-0.2 * self.forkLength))
        turbidity = 0.4703560632 if self.turbidity < 0.4703560632 else self.turbidity
        turbidityAdjustment = (31.64 - 13.31 * np.log10(turbidity)) / 36
        # self.status("Reaction distance of {0:.2f} cm for prey type of mean length {1:.2f}.".format(baseReactionDistance, preyType.length))
        return baseReactionDistance * turbidityAdjustment * self.reactionDistanceMultiplier

    @functools.lru_cache(maxsize=2048)
    def maximumCaptureDistance(self, preyType, waterVelocity):
        """ Maximum distance (measured in cm in the plane perpendicular to the focal point) at which the fish can capture prey.  
            Source: Hughes & Dill 1990 * NOTE THAT THIS FUNCTION IS CURRENTLY NOT USED IN THE PROGRAM. Here's why:
            In short, its functionality is redundant with that of the capture success curve below, and there is no mechanistic
            reason to believe maximum capture distance is a more realistic approach. It also relies on the faulty assumption
            that fish capture prey at their maximum sustainable swimming speed (they usually go slower), which is inconsistent
            with the model's more realistic assumption that they swim at a speed close to the water velocity to capture prey,
            as used when actually computing capture costs and time."""
        rd = self.reactionDistance(preyType)
        return np.sqrt(rd ** 2 - (waterVelocity * rd / self.maximumSustainableSwimmingSpeed) ** 2)

    @functools.lru_cache(maxsize=32768)
    def captureSuccess(self, preyType, waterVelocity, preyDistance):
        """" Logistic regression from Rosenfeld & Taylor 2009, based on data from Hill & Grossman 1993 """
        V = waterVelocity
        d = preyDistance
        RD = self.reactionDistance(preyType)
        FL = self.forkLength  # in cm
        u = 1.28 - 0.0588 * V + 0.383 * FL - 0.0918 * (d / RD) - 0.210 * V * (d / RD)
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
            water)."""
        pursuitDistance = (2 / 3) * self.reactionDistance(preyType)
        pursuitTime = pursuitDistance / preyVelocity
        returnDistance = pursuitDistance  # assumption, for now
        returnTime = returnDistance / self.optimalVelocity  # return time is not factored into "handling time", but it is counted toward maneuver unsteady swimming costs
        unsteadyPursuitVelocity = np.sqrt(3.0 * preyVelocity ** 2)  # effective velocity used for pursuit cost to account for unstady swimming, Hayes et al 2016 eqn 8
        unsteadyReturnVelocity = np.sqrt(3.0 * self.optimalVelocity ** 2)  # effective velocity used for return cost to account for unstady swimming, Hayes et al 2016 eqn 8
        # Note turnCostFactor was edited 7/26/19 to correct for a typo in Hayes et al 2016; but a new typo introduced here (2.2665 instead of 0.022665) was corrected 3/10/2020
        turnCostFactor = 0.9601 * np.exp(0.022665 * preyVelocity)  # factor in the additional cost of turning beyond that of unsteady swimming, Hayes et al 2016 eqn 9, with cm-to-m conversion 0.01
        swimmingCost = (pursuitTime * self.swimmingCost(unsteadyPursuitVelocity) + returnTime * self.swimmingCost(unsteadyReturnVelocity)) * turnCostFactor
        # self.status("Individual maneuver has swimming cost {0:.2f} based on swimming {3:.2f} s at unsteady velocity {1:.2f} for velocity {2:.2f} with turn cost factor {4:.2f}.".format(swimmingCost,unsteadyVelocity,gridCell['velocity'],totalTime,turnCostFactor))
        return (pursuitTime, swimmingCost)  # returned tuple contains the "handling time" (s) and energy cost (J) of one maneuver

    @functools.lru_cache(maxsize=2048)
    def swimmingCost(self, velocity):
        """ This function calls out to the selected swimming cost model. """
        if self.swimmingCostSubmodel == 0:
            return self.swimmingCostHayesEtAl(velocity * self.focalVelocityScaler)
        elif self.swimmingCostSubmodel == 1:
            return self.swimmingCostHayesRainbow(velocity * self.focalVelocityScaler)
        elif self.swimmingCostSubmodel == 2:
            return self.swimmingCostTrudelWelchSockeye(velocity * self.focalVelocityScaler)
        elif self.swimmingCostSubmodel == 3:
            return self.swimmingCostTrudelWelchCoho(velocity * self.focalVelocityScaler)
        elif self.swimmingCostSubmodel == 4:
            return self.swimmingCostTrudelWelchChinook(velocity * self.focalVelocityScaler)

    def swimmingCostHayesEtAl(self, velocity):
        """ Based on Hayes et al 2016, which is based mainly on parameters for brown trout from Elliott (1976) and rainbow trout from 
            Rand et al (1993). Note that this equation appears correctly in Hayes et al 2016, but an incorrect version of the same equation
            with a typo appears in some previous papers. It gives swimming cost (J/s) based on velocity (m/s in their paper, converted
            here from cm/s), mass (g), and temperature (C)."""
        a = 4.126 if self.waterTemperature < 7.1 else 8.277
        b1 = 0.734 if self.waterTemperature < 7.1 else 0.731
        b2 = 0.192 if self.waterTemperature < 7.1 else 0.094
        b3 = 2.34
        return a * self.mass ** b1 * np.exp(b2 * self.waterTemperature) * np.exp(b3 * 0.01 * velocity) * 4.1868 / 86400

    def swimmingCostHayesRainbow(self, velocity):
        """ An updated version of Hayes et al. (2016) with parameters for rainbow trout, used in Dodrill et al. (2016)."""
        RA = 0.013
        RB = -0.217
        RQ = 2.2
        RTO = 22
        RTM = 26
        Y = np.log(RQ) * (RTM - RTO + 2)
        Z = np.log(RQ) * (RTM - RTO)
        X = (Z ** 2) * (1 + (1 + 40 / Y) ** 0.5) ** 2 / 400
        V = (RTM - self.waterTemperature) / (RTM - RTO)
        Rs_FT = V ** X * np.exp(X * (1 - V))
        Rs = RA * (self.mass ** RB) * Rs_FT * self.mass * 3240  ## SMR (e.g., costs at 0 velocity) - cal/day
        SC = (Rs * np.exp(0.0234 * velocity)) / 24  ## swimming costs per hour - velocity input is in cm/s
        return SC * 4.184 * (1 / 3600.0)  ## Return swimming cost in Joules per second

    def swimmingCostTrudelWelchSockeye(self, velocity):
        """the remaining functions are from Trudel and Welch (2005), who use an additive approach to model the energy costs
        of swimming (additional to SMR)
        """
        oq = 14.1  # oxycaloric equivalent in units (j*mgO2) taken as 14.1 from Videler 1993
        smr = (1 / 3600.0) * oq * np.exp(-2.94 + (0.87 * np.log(self.mass)) + (0.064 * self.waterTemperature))  ## SMR
        sc = (1 / 3600.0) * oq * np.exp(-6.25 + (0.72 * np.log(self.mass)) + (1.60 * np.log(velocity)))  ## Swimming costs
        return (sc + smr)

    def swimmingCostTrudelWelchCoho(self, velocity):
        """Calculates swimming cost based on sockeye parameters from regression from Trudel and Welch (2005). Then applies empirical ratios to approximate coho swimming costs and SMR"""
        oq = 14.1  # oxycaloric equivalent in units (j*mgO2) taken as 14.1 from Videler 1993
        smr = (1 / 3600.0) * oq * np.exp(-2.94 + (0.87 * np.log(self.mass)) + (0.064 * self.waterTemperature))  ## SMR
        sc = (1 / 3600.0) * oq * np.exp(-6.25 + (0.72 * np.log(self.mass)) + (1.60 * np.log(velocity)))  ## Swimming costs
        return (sc + (smr * 1.75))

    def swimmingCostTrudelWelchChinook(self, velocity):
        oq = 14.1  # oxycaloric equivalent in units (j*mgO2) taken as 14.1 from Videler 1993
        smr = (1 / 3600.0) * oq * np.exp(-2.94 + (0.87 * np.log(self.mass)) + (0.064 * self.waterTemperature))  ## SMR
        sc = (1 / 3600.0) * oq * np.exp(-6.25 + (0.72 * np.log(self.mass)) + (1.60 * np.log(velocity)))  ## Swimming costs
        return (sc * 0.73) + (smr * 1.15)

    @functools.lru_cache(maxsize=2048)
    def maxDailyConsumption(self, whichConsumptionParameters):
        """ This abstracts the daily consumption portion of the bioenergetics model out of the proportionOfEnergyAssimilated function
            below because it is also used elsewhere."""
        if whichConsumptionParameters == 0:  # Parameter values are for Coho and Chinook from Fish Bioenergetics 3.0 manual and Stewart and Ibarra 1992
            CTO = 15  # Water temperature corresponding to 0.98 of the maximun consumption rate
            CTM = 18  # Water temperature at which temperature dependence is still 0.98 of the maximum
            CTL = 24  # Upper water temperature at which temperature dependence is some reduced fraction (CK4) of the maximum rate
            CQ = 5  # Lower water temperature at which temperature dependence is some reduced fraction (CK1) of the maximum rate
            CK1 = 0.36  # Unitless fraction, see above
            CK4 = 0.01  # Unitless fraction, see above
        elif whichConsumptionParameters == 1:  # Parameters for rainbow trout from Railsback and Rose 1999 and Rand 1993
            CTO = 25  # Water temperature corresponding to 0.98 of the maximun consumption rate
            CTM = 22.5  # Water temperature at which temperature dependence is still 0.98 of the maximum
            CTL = 24.3  # Upper water temperature at which temperature dependence is some reduced fraction (CK4) of the maximum rate
            CQ = 3.5  # Lower water temperature at which temperature dependence is some reduced fraction (CK1) of the maximum rate
            CK1 = 0.2  # Unitless fraction, see above
            CK4 = 0.2  # Unitless fraction, see above
        # Equations for consumption temperature dependence based on equation set 3 on page 2-4 of Fish Bioenergetics 3.0 manual
        G1 = (1 / (CTO - CQ)) * np.log((0.98 * (1 - CK1)) / (CK1 * 0.02))
        G2 = (1 / (CTL - CTM)) * np.log((0.98 * (1 - CK4)) / (CK4 * 0.02))
        L1 = np.exp(G1 * (self.waterTemperature - CQ))
        L2 = np.exp(G2 * (CTL - self.waterTemperature))
        Ka = (CK1 * L1) / (1 + CK1 * (L1 - 1))
        Kb = (CK4 * L2) / (1 + CK4 * (L2 - 1))
        f_of_T = Ka * Kb  # temperature dependence function for consumption
        # Basic form of consumption function from box on page 2-2 of Fish Bioenergetis 3.0 manual
        CA = 0.303  # intercept of the allometric mass function for a 1 g fish at optimum water temperature
        CB = -0.275  # slope of allometric mass function, i.e. coefficient of mass dependence
        Cmax = CA * (self.mass ** CB) * f_of_T  # maximum specific feeding rate (g /g /day)
        return Cmax

    @functools.lru_cache(maxsize=2048)
    def specificConsumptionRate(self, energyIntakeRate, hours=24):
        """ Returns specific consumption rate in units of g/g/(hours) based on energy intake rate input in J/s.
            If 'hours' is 24, this is the variable 'C' from bioenergetics models. However, for daily calculations,
            we use hours=1 to get consumption rate in g/g/hour and then add it to g/g/day across hours with varying
            intake rates."""
        return (energyIntakeRate / 3626) * (60 * 60 * hours) / self.mass

    @functools.lru_cache(maxsize=2048)
    def proportionOfEnergyAssimilated(self, energyIntakeRate):
        """ Calculates the proportion of the caloric content of the food source that can actually be assimilated and available for growth or other needs to
             the fish. The input energyIntakeRate should be in J/s, and needs to be converted in this function to something else."""
        assimilationMethod = self.ui.cbAssimilationMethod.currentIndex()
        if assimilationMethod == 0:
            return 0.6  # Value from Tucker and Rasmussen 1999 and Hewett and Johnson 1992
        elif assimilationMethod == 1:
            return 0.7  # Value from Elliott 1982 via Hughes et al 2003
        elif assimilationMethod in (2, 3):
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
            C = self.specificConsumptionRate(energyIntakeRate, 24)
            Cmax = self.maxDailyConsumption(assimilationMethod - 2)  # subtracting 2 just maps to the correct models given how the indices are arranged in the dropdown boxes
            if C > Cmax: C = Cmax  # IMPORTANT ASSUMPTION -- A fish CAN feed at an instantaneous NREI exceeding Cmax. For assimilation, we assume it doesn't feed above Cmax for the day, i.e. it eventually stops feeding, and we cap C at Cmax. This avoids errors with negative excretion/SDA, etc.
            p = C / Cmax  # proportion of maximum consumption
            # Parameter values below are for Coho and Chinook from Fish Bioenergetics 3.0 manual and Stewart and Ibarra 1992
            # Parameters for egestion and excretion based on Equation Set 2 on page 2-7 (clarified from Elliott 1976)
            FA = 0.212  # Intercept of the proportion of consumed energy egested versus water temperature and ration
            FB = -0.222  # Coefficient of water temperature dependence of egestion
            FG = 0.631  # Coefficient for feeding level dependence of egestion
            UA = 0.0314  # These three are the same
            UB = 0.58  # as the above three, but for
            UG = -0.299  # excretion, not egestion
            F = FA * (self.waterTemperature ** FB) * np.exp(FG * p) * C  # egestion, g /g /day
            U = UA * (self.waterTemperature ** UB) * np.exp(UG * p) * (C - F)  # excretion, g /g /day
            # Specific dynamic action (energy spent digesting prey)
            SDA = 0.172  # unitless coefficient for specific dynamic action
            S = SDA * (C - F)  # S is the assimilated energy lost to specific dynamic action, from page 2-5
            proportionAssimilated = (C - (F + U + S)) / C if C > 0 else 0  # proportion of consumed calories assimilated and available for respiration or growth
            if C > 0 and not 0 < proportionAssimilated < 1:
                self.ui.status("Warning, bad assimilation: with C = {0:8.4f}, F = {1:8.4f}, U = {2:8.4f}, S = {3:8.4f}, and p = {5:4.4f}, fish is assimilating {4:4.4f}".format(C, F, U, S, proportionAssimilated, p))
            return proportionAssimilated

    def clear_caches(self):
        """ This needs to be run anytime a property that affects one of the cached functions, such as mass or temperature, is adjusted. """
        self.reactionDistance.cache_clear()
        self.focalDepth.cache_clear()
        self.swimmingCost.cache_clear()
        self.handlingStats.cache_clear()
        self.proportionOfEnergyAssimilated.cache_clear()
        self.captureSuccess.cache_clear()
        self.maxDailyConsumption.cache_clear()
        self.specificConsumptionRate.cache_clear()
        self.preyDetectionProbability.cache_clear()

    def runForagingModel(self, waterDepth, meanColumnVelocity, shouldOptimizeDiet, gridSize=10, transectInterpolations=None, hour=None):
        """ This wrapper function simply calls the correct function from the two below based on whether diet optimization
            is allowed (which is more computationally expensive) or not."""
        if shouldOptimizeDiet:
            return self.runForagingModelWithDietOptimization(waterDepth, meanColumnVelocity, gridSize, transectInterpolations, hour)
        else:
            return self.runForagingModelWithFixedDiet(waterDepth, meanColumnVelocity, gridSize, transectInterpolations, hour)

    def runForagingModelWithFixedDiet(self, waterDepth, meanColumnVelocity, gridSize, transectInterpolations, hour):
        """ Calculates net rate of energy intake and lots of other internal/diagnostic measures. The 'total' variables
            calculated here are totals across all prey types and grid cells per unit (second) of searching time.
             """
        if waterDepth <= 0 or meanColumnVelocity <= 0:
            return EmptySingleModelResult(waterDepth, meanColumnVelocity, self.preyTypes)
        if self.turbulenceAdjustment == 0:  # No turbulence adjustment applied
            totalFocalSwimmingCost = self.swimmingCost(CalculationGrid.velocityAtDepth(self.velocityProfileMethod, self.focalDepth(waterDepth), waterDepth, meanColumnVelocity, self.roughness))
        elif self.turbulenceAdjustment == 1:  # Webb (1991) factor applied to increase costs due to unsteady focal swimming in turbulent flows
            totalFocalSwimmingCost = self.swimmingCost(np.sqrt(3 * CalculationGrid.velocityAtDepth(self.velocityProfileMethod, self.focalDepth(waterDepth), waterDepth, meanColumnVelocity, self.roughness) ** 2))
        totalEnergyIntake = 0
        totalCaptureManeuverCost = 0
        totalHandlingTime = 0
        totalPreyIngested = 0
        totalPreyEncountered = 0
        totalReactionDistance = 0
        for preyType in self.preyTypes:
            if transectInterpolations == None:
                grid = CalculationGrid(self.reactionDistance(preyType), self.focalDepth(waterDepth), waterDepth, meanColumnVelocity, self.velocityProfileMethod, gridSize, self.roughness)
            else:
                grid = TransectCalculationGrid(transectInterpolations, self.positionOnTransect, self.reactionDistance(preyType), self.focalDepth(waterDepth), self.velocityProfileMethod, gridSize)
            preyType.ingestionCount = 0
            for cell in grid.cells:
                cell.captureSuccess = self.preyDetectionProbability(hour) * self.captureSuccess(preyType, cell.velocity, cell.distance)
                cell.encounterRate = grid.symmetryFactor * cell.area * cell.velocity * (self.hourlyDriftMultiplier * preyType.driftDensity * 1e-6)  # 1e-6 converts prey/m^3 to prey/cm^3
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
        return SingleModelResult(waterDepth, meanColumnVelocity, self.preyTypes, totalHandlingTime, totalAssimilableEnergyIntake, totalReactionDistance, totalPreyEncountered, totalPreyIngested, totalCaptureManeuverCost, totalFocalSwimmingCost, proportionAssimilated)

    def runForagingModelWithDietOptimization(self, waterDepth, meanColumnVelocity, gridSize, transectInterpolations, hour):
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
                nreiWithPreyType = self.runForagingModelWithFixedDiet(waterDepth, meanColumnVelocity, gridSize, transectInterpolations, hour).netRateOfEnergyIntake
                self.preyTypes.remove(preyType)  # temporarily remove the prey type
                nreiWithoutPreyType = self.runForagingModelWithFixedDiet(waterDepth, meanColumnVelocity, gridSize, transectInterpolations, hour).netRateOfEnergyIntake
                if nreiWithPreyType > nreiWithoutPreyType: self.preyTypes.insert(0, preyType)  # put it back in if it was beneficial
            endPreyTypeCount = len(self.preyTypes)
        result = self.runForagingModelWithFixedDiet(waterDepth, meanColumnVelocity, gridSize, transectInterpolations, hour)
        self.preyTypes = deepcopy(originalPreyTypes)  # Restore the forager's prey type options for the next model run
        return result

    def predictedIllumination(self, hourOfDay):
        """ Computes illumination in klux
        Gets date as user input and time from the timeOfDay parameter
        """
        #todo improve comment documentation and function name here and for detection prob wilzbach below
        latitude_deg = float(self.ui.leLatitude.text())  # positive in the northern hemisphere
        longitude_deg = float(self.ui.leLongitude.text())  # negative reckoning west from prime meridian in Greenwich, England
        monthAndDay = self.ui.deMonthAndDay.date()
        timezone_name = TimezoneFinder().timezone_at(lng=longitude_deg, lat=latitude_deg)
        timezone_object = pytz.timezone(timezone_name)
        date = datetime.datetime(2000, monthAndDay.month(), monthAndDay.day(), hourOfDay, 0, 0, 0, tzinfo=timezone_object)
        solar_altitude_deg = get_altitude(latitude_deg, longitude_deg, date)
        irradiation_wm2 = get_radiation_direct(date, solar_altitude_deg)  # direct solar irradiation in watts/m2
        # Note that 1000 W/m2 equals approximately 120,000 lux according to https://ieee-dataport.org/open-access/conversion-guide-solar-irradiance-and-lux-illuminance#:~:text=Solar%20Irradiance%20of%201%20Sun,m2)%20equals%20approximately%20120%2C000%20Lux.
        # but this relationship can vary quite a bit.
        irradiation_lux = irradiation_wm2 * 120  # applies the conversion above
        return irradiation_lux

    @functools.lru_cache(maxsize=2048)
    def preyDetectionProbability(self, hourOfDay):
        """ Wrapper for consideration of more complex prey detection functions later on; for now it just takes one input
            and uses the light-based function."""
        if hourOfDay is not None:
            return self.basePreyDetectionProbability * self.lightSensitiveDetectionProbability(hourOfDay)
        else:
            return self.basePreyDetectionProbability

    def lightSensitiveDetectionProbability(self, hourOfDay):
        """ Returns the light-sensitive prey capture success from the field experiments of Wilzbach et al 1986 (Fig 4), being
            interpreted here as detection probabilities, and based on large 7-10 mm Culicid prey. The same study found
            detection probabilities substantially lower (by around 25 %) for small (3-5 mm) prey, but did not provide
            a good framework or dataset to combine these effects and model both together. A better model of prey detection
            is sorely needed, but considering light alone in this case is better than nothing.

            This output is constrained to a max of 1 and a minimum set somewhat arbitrarily to 0.1. The regression equation
            from Wilzbach was developed under daylight conditions with varied illumination, and extrapolating it to lower light
            gives an x-intercept (i.e. zero detection probability) of 45.5 lux. Also, the solar illumination model above isn't
            meant to handle brightness after dark. Combined, these models basically don't say what happens when it gets dim/dark,
            yet we know the fish feed at those times (maybe preferring more detectable prey, such as on the surface). So we
            set an arbitrary minimum of 0.1 for the detection rate. The lowest detection rate in the graph from the paper was
            about 0.34 at 1000 lux. With a threshold of 0.1, we are extrapolating Wilzbach's relationship down to 100 lux
            (around the brightness of a dimly lit but navigable stairwell or warehouse) and assuming the minimum beyond there."""
        detection_probability_darkness = float(self.ui.leNighttimeDetectionProbability.text())
        irradiation_lux = self.predictedIllumination(hourOfDay)
        if irradiation_lux <= 0:
            return detection_probability_darkness
        detection_probability_wilzbach = -0.42 + 0.11 * np.log(irradiation_lux)
        return min(max(detection_probability_wilzbach, detection_probability_darkness), 1)

    def runDailyModel(self, depth, velocity, shouldOptimizeDiet, gridSize, transectInterpolations):
        """ This function will run the foraging model at 1-hour intervals for an entire day, based on parameters from the
            Daily Settings tab. """
        # ---------------------------------------------------------------------------------------------------------------
        # Process inputs from the custom hourly details specification file, if any, storing in a dictionary keyed by hour
        # ---------------------------------------------------------------------------------------------------------------
        hourlyDetails = {}
        for hour in range(24):          # First, fill the dictionary with default values (i.e. no effect) for blank entries
            hourlyDetails[hour] = {
                'temperature': self.waterTemperature,
                'forcedRest': False,
                'riskMultiplier': 1,
                'driftMultiplier': 1,
                'customDriftFile': ""
            }
        if self.ui.leHourlyDetailsFile.text() != "" and os.path.exists(self.ui.leHourlyDetailsFile.text()):
            with open(self.ui.leHourlyDetailsFile.text(), newline='') as csvFile:
                reader = csv.reader(csvFile)
                next(reader, None)  # skip the header row
                for row in reader:
                    try:
                        hour = int(row[0])
                        if row[1] != "":
                            hourlyDetails[hour]['temperature'] = float(row[1])
                        if row[2] != "":
                            hourlyDetails[hour]['forcedRest'] = bool(int(row[2]))
                        if row[3] != "":
                            hourlyDetails[hour]['riskMultiplier'] = float(row[3])
                        if row[4] != "":
                            hourlyDetails[hour]['driftMultiplier'] = float(row[4])
                        if row[5] != "":
                            hourlyDetails[hour]['customDriftFile'] = row[5]
                    except ValueError as err:
                        message = "Value encountered in the hourly details file that could not be interpreted! Skipping a row. Error: {0}".format(err)
                        self.ui.statusError(message)
        # ----------------------------------------------------------------------------------------------------------
        # Process other inputs from the user interface for hourly settings
        # ----------------------------------------------------------------------------------------------------------
        specificHoursFeedingIsAllowed = [hour for hour in range(24) if not hourlyDetails[hour]['forcedRest']]
        strategy = self.ui.cbForagingStrategy.currentIndex()
        maxHoursFeedingIsAllowed = float(self.ui.leMaxHoursToFeed.text())
        # Predation risk is input in the interface in terms of easier-to-picture 90-day probabilities of being predated,
        # but that is converted here to actual hourly probabilities, which can then be modified by hour or location.
        baselineHourlyRisk = 1 - (1 - float(self.ui.leBaselineHourlyPredationRiskInTermsOf90DayHorizon.text())) ** (1/(24*90))
        dailyRiskScaleConstant = 1 - (1 - float(self.ui.leRiskScaleConstant.text())) ** (1 / 90)
        # ----------------------------------------------------------------------------------------------------------
        # Compute the results of foraging during every hour of the day in which foraging is allowed.
        # ----------------------------------------------------------------------------------------------------------
        hourlyResults = []
        originalPreyTypes = deepcopy(self.preyTypes)
        self.ui.pbDailyRunProgressHour.setMaximum(23)
        for hour in range(24):
            self.ui.pbDailyRunProgressHour.setValue(hour)
            if hour in specificHoursFeedingIsAllowed:
                self.hourlyDriftMultiplier = hourlyDetails[hour]['driftMultiplier']
                hasCustomPreyTypes = (hourlyDetails[hour]['customDriftFile'] != "")
                if hasCustomPreyTypes:
                    self.filterPreyTypes(PreyType.loadPreyTypes(hourlyDetails[hour]['customDriftFile'], self))
                resultAtHour = self.runForagingModel(depth, velocity, shouldOptimizeDiet, gridSize, transectInterpolations, hour)
                self.ui.status("Ran model at hour {0} with NREI {1}.".format(hour, resultAtHour.netRateOfEnergyIntake))
                resultAtHour.hour = hour
                resultAtHour.hourlyRisk = np.clip(baselineHourlyRisk * hourlyDetails[hour]['riskMultiplier'], 0, 1)  # todo include spatial factor here when implemented
                hourlyResults.append(resultAtHour)
                if hasCustomPreyTypes:
                    self.preyTypes = deepcopy(originalPreyTypes)  # restore the default prey types after each iteration with custom ones
                self.ui.app.processEvents()  # Forces the progress bar and status window to update with each iteration rather than waiting until the end of the loop.
        self.ui.pbDailyRunProgressHour.setValue(0)
        self.hourlyDriftMultiplier = 1  # restore to default value for future calculations
        Cmax = self.maxDailyConsumption(self.ui.cbConsumptionParameters.currentIndex())
        # ----------------------------------------------------------------------------------------------------------
        # Sort the hourly results in descending order of preference according to the fish's strategy.
        # ----------------------------------------------------------------------------------------------------------
        if strategy == 0:  # Efficiency maximizing
            # The efficiency-maximizing fish wants to spend the least energy in order to reach MaxCons, so it feeds in the hours
            # with the best NREI/SC ratio until it reaches MaxCons.
            hourlyResults.sort(key=lambda r: r.netRateOfEnergyIntake / r.totalEnergyCostRate, reverse=True)  # sort descending by NREI/SC
        elif strategy == 1:  # Risk balancing
            hourlyResults.sort(key=lambda r: r.hourlyRisk / r.netRateOfEnergyIntake, reverse=False)  # sort ascending by risk/NREI ratio
        # ----------------------------------------------------------------------------------------------------------
        # Now iterate through the sorted hours (best hours first) totaling the fish's foraging progress until it
        # either fills up, runs out of hours available to forage, or gets into hours with a negative NREI where
        # resting is better than foraging. In all those cases it stops foraging in the subsequent hours.
        # ----------------------------------------------------------------------------------------------------------
        dailyGrossEnergyIntake = 0
        dailySpecificConsumption = 0
        dailyCost = 0
        dailyFocalSwimmingCost = 0
        dailyCaptureManeuverCost = 0
        dailyHoursForaging = 0
        for i, result in enumerate(hourlyResults):
            grossRateOfEnergyIntake = result.grossRateOfEnergyIntake
            focalSwimmingCostRate = result.focalSwimmingCostRate
            captureManeuverCostRate = result.captureManeuverCostRate
            costRate = result.totalEnergyCostRate
            # todo remove maxHourlyRisk
            if i + 1 > maxHoursFeedingIsAllowed or grossRateOfEnergyIntake < costRate or dailySpecificConsumption >= Cmax:
                result.actuallyForaged = False
                # todo add test here for risk metric
                break  # using 'break' instead of 'continue' because hours are already ranked by desirability
            if i == 0:
                riskProductAccumulator = result.risk
                dneiSumAccumulator = result.netRateOfEnergyIntake
                previousRiskMetric = (dailyRiskScaleConstant + riskProductAccumulator) / dneiSumAccumulator
                result.actuallyForaged = True
            else:
                riskProductAccumulator *= result.risk
                dneiSumAccumulator += result.netRateOfEnergyIntake
                currentRiskMetric = (dailyRiskScaleConstant + riskProductAccumulator) / dneiSumAccumulator
                if currentRiskMetric <= previousRiskMetric:
                    result.actuallyForaged = True
                    previousRiskMetric = currentRiskMetric
                else:
                    result.actuallyForaged = False
                    break

            hourlySpecificConsumption = self.specificConsumptionRate(grossRateOfEnergyIntake, hours=1)
            if dailySpecificConsumption + hourlySpecificConsumption <= Cmax:
                dailySpecificConsumption += hourlySpecificConsumption
                dailyCost += costRate * 3600  # convert J/s to J/h and add to daily total
                dailyFocalSwimmingCost += focalSwimmingCostRate * 3600
                dailyCaptureManeuverCost += captureManeuverCostRate * 3600
                dailyGrossEnergyIntake += grossRateOfEnergyIntake * 3600  # convert J/s to J/h and add to daily total
                dailyHoursForaging += 1
            else:
                proportionOfHourForaged = (Cmax - dailySpecificConsumption) / hourlySpecificConsumption
                dailySpecificConsumption = Cmax
                dailyCost += costRate * 3600 * proportionOfHourForaged
                dailyFocalSwimmingCost += focalSwimmingCostRate * 3600 * proportionOfHourForaged
                dailyCaptureManeuverCost += captureManeuverCostRate * 3600 * proportionOfHourForaged
                dailyGrossEnergyIntake += grossRateOfEnergyIntake * 3600 * proportionOfHourForaged
                dailyHoursForaging += proportionOfHourForaged
        # ---------------------------------------------------------------------------------------------------------------
        # Reset the result plot selection dropdown to values for daily rather than instantaneous model runs
        # ---------------------------------------------------------------------------------------------------------------
        self.ui.changePlotOptions(1)
        # ----------------------------------------------------------------------------------------------------------
        # Arrange and return the completed results in a useful form
        # ----------------------------------------------------------------------------------------------------------
        hourlyResults.sort(key=lambda r: r.hour)  # sort back to chronological order for diagnostics
        dailyNetEnergyIntake = dailyGrossEnergyIntake - dailyCost
        dailyRisk = 1 - np.prod([1 - result.hourlyRisk for result in hourlyResults if result.actuallyForaged])
        dailyRiskBalancingMetric = (dailyRiskScaleConstant + dailyRisk) / dailyNetEnergyIntake
        return DailyRunResult(depth, velocity, hourlyResults, dailyNetEnergyIntake, dailyGrossEnergyIntake, dailyCost, dailyHoursForaging, dailyFocalSwimmingCost, dailyCaptureManeuverCost, dailyRisk, dailyRiskBalancingMetric, dailySpecificConsumption, dailySpecificConsumption / Cmax)

    def status(self, message):
        if self.ui is not None:
            self.ui.status(message)
        else:
            print(message)
