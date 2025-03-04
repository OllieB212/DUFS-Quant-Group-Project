#!/usr/bin/env python
import numpy as np
import EuropeanCall
import VolSurface
import SVIParamVol
test = 3
if test == 1:
    #flat vol
    S = 100.0
    K = np.array([90.0, 100.0, 110.0])
    T = 1.0
    r = 0.05
    flatVol = 0.3
    priceBS = EuropeanCall.BlackScholes(S, K, T, r, flatVol)
    rng =  np.random.default_rng(seed=97)
    priceLN = EuropeanCall.MCLogNormal(S, K, T, r, flatVol, 50000, rng)
    print(f"priceBS={priceBS}, priceMCLN={priceLN}")

    #local vol needs a vol surface
    strikesVS = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0]
    maturitiesVS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    volsVS = np.full((len(maturitiesVS), len(strikesVS)), flatVol)
    volSurf = VolSurface.VolSurface(maturitiesVS, strikesVS, volsVS)

    rng =  np.random.default_rng(seed=97)
    priceLV = EuropeanCall.MCLV(S, K, T, r, volSurf, 50000, rng)
    print(f"priceBS={priceBS}, priceMCLV={priceLV}")

elif test == 2:
    #vol no longer flat. Vary in strike dimension
    S = 100.0
    strikes = np.array([80.0, 100.0, 120.0])
    T = 1.0
    r = 0.05
    strikesVS = [40, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0]
    maturitiesVS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    volRow = [0.35, 0.33, 0.32, 0.3, 0.33, 0.37, 0.45]
    volsVS = np.tile(volRow, (len(maturitiesVS), 1))
    volSurf = VolSurface.VolSurface(maturitiesVS, strikesVS, volsVS)

    relevantVols = [0.32, 0.3, 0.33]
    for i in range(0, len(strikes)):
        flatVol = relevantVols[i]
        K = strikes[i]
        priceBS = EuropeanCall.BlackScholes(S, K, T, r, flatVol)
        print(f"For strike={K}, vol={flatVol} priceBS={priceBS}")
    for i in range(0, len(strikes)):
        flatVol = relevantVols[i]
        rng =  np.random.default_rng(seed=97)
        priceLN = EuropeanCall.MCLogNormal(S, strikes, T, r, flatVol, 50000, rng)
        print(f"For vol={flatVol}\n   "\
              f"priceMCLN={priceLN}")
    rng =  np.random.default_rng(seed=97)
    priceLV = EuropeanCall.MCLV(S, strikes, T, r, volSurf, 50000, rng, 3, 400)
    print(f"priceMCLV={priceLV}")

elif test == 3:
    #use SVIParamVol
    S = 100.0
    volSurf = SVIParamVol
test = 3
if test == 1:
    #flat vol
    S = 100.0
    K = np.array([90.0, 100.0, 110.0])
    T = 1.0
    r = 0.05
    flatVol = 0.3
    priceBS = EuropeanCall.BlackScholes(S, K, T, r, flatVol)
    rng =  np.random.default_rng(seed=97)
    priceLN = EuropeanCall.MCLogNormal(S, K, T, r, flatVol, 50000, rng)
    print(f"priceBS={priceBS}, priceMCLN={priceLN}")

    #local vol needs a vol surface
    strikesVS = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0]
    maturitiesVS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    volsVS = np.full((len(maturitiesVS), len(strikesVS)), flatVol)
    volSurf = VolSurface.VolSurface(maturitiesVS, strikesVS, volsVS)

    rng =  np.random.default_rng(seed=97)
    priceLV = EuropeanCall.MCLV(S, K, T, r, volSurf, 50000, rng)
    print(f"priceBS={priceBS}, priceMCLV={priceLV}")

elif test == 2:
    #vol no longer flat. Vary in strike dimension
    S = 100.0
    strikes = np.array([80.0, 100.0, 120.0])
    T = 1.0
    r = 0.05
    strikesVS = [40, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0]
    maturitiesVS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    volRow = [0.35, 0.33, 0.32, 0.3, 0.33, 0.37, 0.45]
    volsVS = np.tile(volRow, (len(maturitiesVS), 1))
    volSurf = VolSurface.VolSurface(maturitiesVS, strikesVS, volsVS)

    relevantVols = [0.32, 0.3, 0.33]
    for i in range(0, len(strikes)):
        flatVol = relevantVols[i]
        K = strikes[i]
        priceBS = EuropeanCall.BlackScholes(S, K, T, r, flatVol)
        print(f"For strike={K}, vol={flatVol} priceBS={priceBS}")
    for i in range(0, len(strikes)):
        flatVol = relevantVols[i]
        rng =  np.random.default_rng(seed=97)
        priceLN = EuropeanCall.MCLogNormal(S, strikes, T, r, flatVol, 50000, rng)
        print(f"For vol={flatVol}\n   "\
              f"priceMCLN={priceLN}")
    rng =  np.random.default_rng(seed=97)
    priceLV = EuropeanCall.MCLV(S, strikes, T, r, volSurf, 50000, rng, 3, 400)
    print(f"priceMCLV={priceLV}")

elif test == 3:
    #use SVIParamVol
    S = 100.0
    volSurf = SVIParamVol.SVIParamVolSurf.exampleSurface(S)
    
    strikes = np.array([80.0, 100.0, 120.0])
    T = 1.0
    r = 0.05

    relevantVols = volSurf.calcSmoothVols(np.array([T]), strikes)
    for i in range(0, len(strikes)):
        flatVol = relevantVols[0][i]
        K = strikes[i]
        priceBS = EuropeanCall.BlackScholes(S, K, T, r, flatVol)
        print(f"For strike={K}, vol={flatVol} priceBS={priceBS}")
    for i in range(0, len(strikes)):
        flatVol = relevantVols[0][i]
        rng =  np.random.default_rng(seed=97)
        priceLN = EuropeanCall.MCLogNormal(S, strikes, T, r, flatVol, 50000, rng)
        print(f"For vol={flatVol}\n   "\
              f"priceMCLN={priceLN}")
    rng =  np.random.default_rng(seed=97)
    priceLV = EuropeanCall.MCLV(S, strikes, T, r, volSurf, 50000, rng,
                                S*0.1, S * 2,
                                0.125, 100)
    print(f"priceMCLV={priceLV}")
    
