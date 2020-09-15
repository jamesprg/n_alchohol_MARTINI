#!/usr/bin/env python

import sys
from optparse import OptionParser
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

def gauss(x, *params):
    A, mu, sigma = params
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def fitGaussian(x,y):
    # Based on http://stackoverflow.com/questions/11507028/fit-a-gaussian-function

    # obtain initial guess for Gaussian parameters

    mu = np.sum(x*y) / np.sum(y)
    sigma = np.sqrt( (np.sum(x**2*y) / np.sum(y)) - mu**2)
    dx = (np.max(x) - np.min(x)) / len(x)
    A = dx * np.sum(y) / np.sqrt(2. * np.pi * sigma**2)
    params = (A, mu, sigma)

#    print "Initial guess: A", params[0], "mu", params[1], "sigma", params[2]

    params, _ = curve_fit(gauss, x, y, p0=params)

#    print "Final guess: A", params[0], "mu", params[1], "sigma", params[2]

    return params

def computeThickness(filename, cut=0.05, showplot=False):

    # Read in density file. Ignore lines starting with '#' or '@', which appear in output of gmx density (http://stackoverflow.com/a/24280221)
#    z,rho = np.loadtxt(filename).T
    z,rho = np.genfromtxt((r for r in open(filename) if not r[0] in ('#','@'))).T
    # divide density into two regions (z<0 and z>0), and for each fit a Gaussian, including only data points that are at least cut% of the maximal value   
    
    # lower leaflet peak (z<0)
    rhomax = np.max(rho[z<0])
    #5% pick here, want larger
    zlower = z[np.logical_and(z<0 , rho > cut * rhomax)]
    rholower = rho[np.logical_and(z<0 , rho > cut * rhomax)]
    paramslower = fitGaussian(zlower,rholower)

    # upper leaflet peak (z>0)
    rhomax = np.max(rho[z>0])
    zupper = z[np.logical_and(z>0 , rho > cut * rhomax)]
    rhoupper = rho[np.logical_and(z>0 , rho > cut * rhomax)]
    paramsupper = fitGaussian(zupper,rhoupper)

    if showplot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot (z,rho, 'b.')
        zl = np.arange(np.min(zlower), np.max(zlower), 0.01)
        ax.plot (zl, gauss(zl, *paramslower), 'r-')
        zu = np.arange(np.min(zupper), np.max(zupper), 0.01)
        ax.plot (zu, gauss(zu, *paramsupper), 'g-')
        ax.axvline(paramslower[1]-2*paramslower[2],ls='--',color='pink')
        ax.axvline(paramsupper[1]+2*paramsupper[2],ls='--',color='pink')
        plt.show()
    
    return paramsupper[1] - paramslower[1]

def main(argv):
    parser = OptionParser(usage="Usage: %prog DENSFILE",description="Calculates bilayer thickness from density profile.")
    parser.add_option("-w", action="store_true", dest="showplot", default=False, help="Show plot [default: %default].")
    parser.add_option("-c", dest="cut", default=0.05, help="Determine percentage of points to cut in gaussian fit [default: %default].")
    (options, args) = parser.parse_args(argv[1:])

    if len(args) != 1:
        print("Need exactly 1 arguments, but received " + str(len(args)) + ":", args, file=sys.stderr)
        return 1

    densfilename = args[0]

    thickness = computeThickness(densfilename,options.cut, options.showplot)
    print(thickness)

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
