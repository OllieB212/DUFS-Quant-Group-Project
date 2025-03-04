#!/usr/bin/env python
import numpy as np
import scipy

class SVIParamVol:
    def __init__(self, a, b, sig, rho, m):
        self.a = a
        self.b = b
        self.sig = sig
        self.rho = rho
        self.m = m

    #calculates variance at requested strikes which must be log(strike/
    #forward)
    def calcVar(self, logReqStrikes):
        return self.a + self.b * (self.rho * (logReqStrikes - self.m) + \
                        np.sqrt(np.square(logReqStrikes - self.m) +
                                self.sig ** 2))
        
#inputs are arrays
class SVIParamVolSurf:
    #Require len(sviParamVols) = len(forwards) = len(maturities)
    def __init__(self, maturities, sviParamVols, forwards):
        self.maturities = maturities
        self.sviParamVols = sviParamVols
        self.forwards = forwards

    #calculates grid of vols at requested strikes/maturities.
    def calcSmoothVols(self, reqMaturities, reqStrikes):
        maturities = self.maturities
        sviParamVols = self.sviParamVols
        #start by getting variance at our expiries
        varsAtExpiries = []
        for maturity, volRow, forward in zip(maturities, sviParamVols,
                                             self.forwards):
            if maturity < reqMaturities[-1]:
                logReqStrikes = np.log(reqStrikes/forward)
                varsAtExpiries.append(volRow.calcVar(logReqStrikes))
            else:
                break
        #then add in final row (if needed)
        if len(varsAtExpiries) < len(sviParamVols):
            varsAtExpiries.append(
                sviParamVols[len(varsAtExpiries)].calcVar(logReqStrikes))

        #then calculate vols - start with simple linear interp in var
        volsToCalc = np.empty((len(reqMaturities), len(logReqStrikes)))
        for iMat in range(0, len(reqMaturities)):
            reqMat = reqMaturities[iMat]
            index = np.searchsorted(maturities, reqMat)
            if index == 0:
                varsAtMat = varsAtExpiries[0]*reqMat/maturities[0]
            elif index == len(maturities):
                varsAtMat = varsAtExpiries[-1]*reqMat/maturities[-1]
            else:
                varsAtExpiry = varsAtExpiries[index-1]
                varsAtNextExpiry = varsAtExpiries[index]
                varsAtMat = varsAtExpiry + (reqMat - maturities[index-1]) * \
                    ((varsAtNextExpiry-varsAtExpiry)/(maturities[index] -
                                                      maturities[index-1]))
            volsToCalc[iMat] = np.sqrt(varsAtMat/reqMat)
        return volsToCalc
       
    def exampleSurface(spot):
        maturities = []
        sviParamVols = []
        forwards = []
        maturities.append(0.0055)
        forwards.append(spot)
        sviParamVols.append(SVIParamVol(-0.0001449630, 0.009296544, 0.01967133, -0.2941176, -0.005427323))
        maturities.append(0.1)
        forwards.append(spot)
        sviParamVols.append(SVIParamVol(-0.0008321340,0.024439766,0.06986945,-0.2999753,0.026483640))
        maturities.append(0.18)
        forwards.append(spot)
        sviParamVols.append(SVIParamVol(-0.0008676750,0.028290645,0.08738356,-0.2892204,0.059270300))
        maturities.append(0.25)
        forwards.append(spot)
        sviParamVols.append(SVIParamVol(-0.0000591593,0.033179082,0.08128724,-0.3014043,0.065254921))
        maturities.append(0.5)
        forwards.append(spot)
        sviParamVols.append(SVIParamVol(0.0011431940,0.046279644,0.10406830,-0.3530782,0.094200077))
        maturities.append(0.75)
        forwards.append(spot)
        sviParamVols.append(SVIParamVol(0.0022640980,0.056260415,0.13053393,-0.4387409,0.111123069))
        maturities.append(1.25)
        forwards.append(spot)
        sviParamVols.append(SVIParamVol(0.0040335530,0.073370755,0.17079476,-0.4968970,0.149660916))
        maturities.append(1.75)
        forwards.append(spot)
        sviParamVols.append(SVIParamVol(0.0034526910,0.091723054,0.22368141,-0.4942213,0.185412849))
        volSurf = SVIParamVolSurf(maturities, sviParamVols, forwards)
        return volSurf
    
def test():
    volSurf = SVIParamVolSurf.exampleSurface(100)
    reqStrikes = np.array([10, 50, 80, 95, 105, 180])
    reqMaturities = np.array([0.1, 0.5, 0.8, 1.25, 2.0])
    volGrid1 = volSurf.calcSmoothVols(reqMaturities, reqStrikes)
    print("volGrid1:\n", volGrid1)

if __name__ == "__main__":
    test()    
    
