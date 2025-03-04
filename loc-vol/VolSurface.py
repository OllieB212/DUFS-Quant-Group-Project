#!/usr/bin/env python
import numpy as np
import scipy

class VolSurface:
    #vols needs to be numpy array
    def __init__(self, maturities, strikes, vols): #vols[mat][[strike]
        self.strikes = strikes
        self.maturities = maturities
        self.vols = vols

    #calculates grid of vols at requested strikes/maturities. Linear
    #interpolation in strike dimension, linear in variance in time dimension
    #reqMaturities and reqStrikes must be increasing
    def calcVols(self, reqMaturities, reqStrikes):
        maturities = self.maturities
        strikes = self.strikes
        vols = self.vols
        volsToCalc = np.empty((len(reqMaturities), len(reqStrikes)))
        for iMat in range(0, len(reqMaturities)):
            reqMat = reqMaturities[iMat]
            index = np.searchsorted(maturities, reqMat)
            if index == 0:
                volsAtMat = vols[0] #flat interpolation
            elif index == len(maturities):
                volsAtMat = vols[-1] #flat interpolation
            else:
                varsAtIndex = vols[index-1]**2 * maturities[index-1]
                varsAtNextIndex = vols[index]**2 * maturities[index]
                varsAtMat = varsAtIndex + (reqMat - maturities[index-1]) * \
                    ((varsAtNextIndex-varsAtIndex)/(maturities[index] -
                                                    maturities[index-1]))
                volsAtMat = np.sqrt(varsAtMat/reqMat)

            #now interpolate linearly in strike. Use flat interpolation off
            #end
            volsToCalc[iMat] = np.interp(reqStrikes, strikes, volsAtMat)
        return volsToCalc

    #calculates grid of vols at requested strikes/maturities. Cubic spline
    #interpolation in strike dimension. Cubic spline in variance in
    #time dimension
    #reqMaturities and reqStrikes must be increasing
    def calcSmoothVols(self, reqMaturities, reqStrikes):
        maturities = self.maturities
        strikes = self.strikes
        vols = self.vols
        #have to jump through hoops to get numpy to mutiply each vol by the
        #corresponding maturity
        variance = (vols ** 2) * np.transpose(np.tile(maturities,
                                                      (len(strikes),1)))
        #spline in time dimension
        timeSpline = scipy.interpolate.CubicSpline(
            maturities, variance, bc_type = 'clamped', extrapolate = False)
        varAtReqMat = timeSpline(reqMaturities)
        #turn back to vols
        volAtReqMat = np.sqrt(varAtReqMat/\
            np.transpose(np.tile(reqMaturities,(len(strikes),1))))
        #handle reqMaturities outside of spline (np.where doesn't do the
        #the right thing with the arrays)
        volAtReqMat = [xv if c else yv
                       for c,xv,yv in zip(reqMaturities < maturities[0],
                                          np.tile(vols[0],
                                                  (len(reqMaturities),1)),
                                          volAtReqMat)]
        volAtReqMat = [xv if c else yv
                       for c,xv,yv in zip(reqMaturities > maturities[-1],
                                          np.tile(vols[-1],
                                                  (len(reqMaturities),1)),
                                          volAtReqMat)]
        #then spline in strike dimension
        strikeSpline = scipy.interpolate.CubicSpline(
            strikes, volAtReqMat, axis = 1,
            bc_type = 'clamped', extrapolate = False)
        volsToCalc = strikeSpline(reqStrikes
                                  )
        #handle reqStrikes outside of spline. Easier to handle if we
        #tranpose the array of arrays
        volsToCalc = np.transpose(volsToCalc)
        volsToCalc = [xv if c else yv
                       for c,xv,yv in zip(reqStrikes < strikes[0],
                                          np.tile(strikeSpline(strikes[0]),
                                                  (len(reqStrikes),1)),
                                          volsToCalc)]
        volsToCalc = [xv if c else yv
                       for c,xv,yv in zip(reqStrikes > strikes[-1],
                                          np.tile(strikeSpline(strikes[-1]),
                                                  (len(reqStrikes),1)),
                                          volsToCalc)]
        #and switch it back
        volsToCalc = np.transpose(volsToCalc)
       
        return volsToCalc

    #returns [first strike, last strike] or [only strike] if only one
    def strikeBounds(self):
        if len(self.strikes) == 1:
            return self.strikes[0]
        return [self.strikes[0], self.strikes[-1]]

def test():
    #test out
    #strikes = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    strikes = np.array([90, 100, 110, 120])
    #maturities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    maturities = np.array([0.5, 1.0, 1.5])
    vols = np.full((len(maturities), len(strikes)), 0.3)
    vols[1] = [0.5, 0.4, 0.6, 0.7]
    vols[2] = [0.55, 0.45, 0.65, 0.75]
    volSurf = VolSurface(maturities, strikes, vols)
    volGrid1 = volSurf.calcVols(maturities, strikes)
    volGrid2 = volSurf.calcSmoothVols(maturities, strikes)
    np.set_printoptions(suppress=True)
    print("volGrid1:\n", volGrid1)
    print("volGrid2:\n", volGrid2)
    print("volGrid1 - volGrid2:\n", volGrid1-volGrid2)
    reqStrikes = np.array([10, 50, 95, 105, 180])
    reqMaturities = np.array([0.1, 0.5, 0.8, 1.25, 2.0])
    volGrid1 = volSurf.calcVols(reqMaturities, reqStrikes)
    volGrid2 = volSurf.calcSmoothVols(reqMaturities, reqStrikes)
    print("volGrid1:\n", volGrid1)
    print("volGrid2:\n", volGrid2)
    print("volGrid1 - volGrid2:\n", volGrid1-volGrid2)

    vols = np.full((len(maturities), len(strikes)), 0.3)
    volSurf = VolSurface(maturities, strikes, vols)
    volGrid1 = volSurf.calcVols(reqMaturities, reqStrikes)
    volGrid2 = volSurf.calcSmoothVols(reqMaturities, reqStrikes)
    print("volGrid1:\n", volGrid1)
    print("volGrid2:\n", volGrid2)
    print("volGrid1 - volGrid2:\n", volGrid1-volGrid2)

if __name__ == "__main__":
    test()    
