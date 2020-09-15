
"""
Calculates the number of bilayer crossing events for alcohol and cholesterol in a model MARTINI membranes
"""
import sys
import os
from itertools import islice
from optparse import OptionParser
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import MDAnalysis
import itertools
sys.path.insert(0, r'/Download/')
import msmbuildertools as msm
import bilayertest as comthick
import gaussparams




# argument for gaussian cutoff! Default is 0.05, needs roughly 25% for butanol check your widths and adjust accordingly

def main(argv):
    parser = OptionParser(usage="Usage: %prog [options] GROFILE XTCFILE",description="Plots z-position of phosphate and cholesterol oxygen atoms.")
    parser.add_option("--startframe", dest="startframe", type="int", default=1, help="Start analysis at this frame [default: %default]")
    parser.add_option("--stopframe", dest="stopframe", type="int", default=None, help="Stop analysis at this frame [default: %default]")
    parser.add_option("--stepframe", dest="stepframe", type="int", default=1, help="Step size when iterating over frames [default: %default]")
    parser.add_option("-w", action="store_true", dest="showplot", default=False, help="Show plot [default: %default].")
    (options, args) = parser.parse_args(argv[1:])
    if len(args) < 4:
        print("Need at least 4 arguments, but received " + str(len(args)) + ":", args, file=sys.stderr)
        print("\n python flipflop.py grofilename densfilename alcohol_atomtype gaussian_cutoff")
        return 1
    grofilename = args[0]
    xtcfilename = grofilename[:-3]+'xtc'
    densfilename = args[1]
    alchead = args[2]
    if 'lo' in grofilename:
        phase='lo'
    elif 'ld' in grofilename:
        phase='ld'
    cut=args[3]
    filename=grofilename[:-8]+phase+'.txt'
    cut=float(cut)
    params=gaussparams.computer(densfilename,cut,options.showplot)
    check=comthick.computeThickness(densfilename,cut,options.showplot)
    thickness = params[1][1]-params[0][1]
    print(params)
    print(thickness)
    upcent=params[1][1]
    dncent=params[0][1]
    u = MDAnalysis.Universe(grofilename, xtcfilename)
    atomgrp = u.select_atoms("name "+str(alchead))
    print("Tracking z position of {} atoms...".format(len(atomgrp)))
    zs = np.vstack([(atomgrp.positions[:,2]-_.dimensions[2]/2)/10. for _ in islice(u.trajectory, options.startframe-1, options.stopframe, options.stepframe)])
    print(" read", zs.shape[0], "frames.")
    
    cholgrp = u.select_atoms("name ROH")
    print("Tracking z position of {} atoms...".format(len(cholgrp)), end=' ')
    zsc = np.vstack([(cholgrp.positions[:,2]-_.dimensions[2]/2)/10. for _ in islice(u.trajectory, options.startframe-1, options.stopframe, options.stepframe)])
    print(" read", zsc.shape[0], "frames.")
    
    #for ld
    if phase == 'ld':
        n=1
        print('Ld!')
    #for lo
    if phase == 'lo':
        n=2
        print('Lo!')
       
    regions_alc = -np.ones_like(zs,dtype=int)
    #region 0 is the lower leaflet, with widths plus and minus one standard deviation
    regions_alc[np.logical_and(zs > dncent-n*params[0][2], zs < dncent+n*params[0][2])] = 0

    regions_alc[np.logical_and(zs > -(n*params[1][2]+n*params[0][2])/2, zs < (n*params[1][2]+n*params[0][2])/2)] = 1 
    
    regions_alc[np.logical_and(zs > upcent-n*params[1][2], zs < upcent+n*params[1][2])] = 2 
    regions_alc[np.logical_and(zs > upcent+3*params[1][2], zs < thickness*2)] = 3
    regions_alc[np.logical_and(zs > -(thickness*2), zs < dncent-3*params[0][2]) ] = 4

    
    regions_chol = -np.ones_like(zsc,dtype=int)
    regions_chol[np.logical_and(zsc > dncent-n*params[0][2], zsc < dncent+n*params[0][2])] = 0
    regions_chol[np.logical_and(zsc > -(n*params[1][2]+n*params[0][2])/2, zsc < (n*params[1][2]+n*params[0][2])/2)] = 1 
    regions_chol[np.logical_and(zsc > upcent-n*params[1][2], zsc < upcent+n*params[1][2])] = 2 
  
####adding
    count=1
    flipmat=[]
    flipmat_tb=[]
    flipmat_bt=[]
    flipmat_st=[]
    flipmat_ts=[]
    flipmat_bs=[]
    flipmat_sb=[]

    for r in regions_alc.T:
        r = msm.tba(r)
        s = msm.rlencode(r)[1]
        flips = np.sum(np.alltrue(np.vstack((s[:-2],s[1:-1],s[2:])).T == np.array([0,1,2]),axis=1))+np.sum(np.alltrue(np.vstack((s[:-2],s[1:-1],s[2:])).T == np.array([2,1,0]),axis=1))
        flips_tb = np.sum(np.alltrue(np.vstack((s[:-2],s[1:-1],s[2:])).T == np.array([2,1,0]),axis=1))
        flips_bt = np.sum(np.alltrue(np.vstack((s[:-2],s[1:-1],s[2:])).T == np.array([0,1,2]),axis=1))
        flips_st = np.sum(np.alltrue(np.vstack((s[:-2],s[1:-1])).T == np.array([3,2]),axis=1))
        flips_ts = np.sum(np.alltrue(np.vstack((s[:-2],s[1:-1])).T == np.array([2,3]),axis=1))
        flips_bs = np.sum(np.alltrue(np.vstack((s[:-2],s[1:-1])).T == np.array([0,4]),axis=1))
        flips_sb = np.sum(np.alltrue(np.vstack((s[:-2],s[1:-1])).T == np.array([4,0]),axis=1))
        flipmat_tb.append(flips_tb)
        flipmat_bt.append(flips_bt)
        flipmat_st.append(flips_st)
        flipmat_ts.append(flips_ts)
        flipmat_bs.append(flips_bs)
        flipmat_sb.append(flips_sb)
        flipmat.append(flips)
        count+=1   
    print('Alc Up or down Transition:', sum(flipmat))
    print('Flip Rate:', sum(flipmat)/(u.trajectory.totaltime*atomgrp.n_atoms)*1000**4,'inverse s')
    print('T1 - Top to Bottom (2->0)', sum(flipmat_tb))
    print('T2 - Bottom to Top (0->2)', sum(flipmat_bt))
    print('T3 - Solvent to Top (3->2)', sum(flipmat_st))
    print('T4 - Top to Solvent (2->3)', sum(flipmat_ts))
    print('T5 - Bottom to Solvent (0->4)', sum(flipmat_bs))
    print('T6 - Solvent to Bottom (4->0)', sum(flipmat_sb))

####### 
    count=1
    cflipmat=[]
    cflipmat_tb=[]
    cflipmat_bt=[]
    for r in regions_chol.T:
        r = msm.tba(r)
        s = msm.rlencode(r)[1]
        flips = np.sum(np.alltrue(np.vstack((s[:-2],s[1:-1],s[2:])).T == np.array([0,1,2]),axis=1))+np.sum(np.alltrue(np.vstack((s[:-2],s[1:-1],s[2:])).T == np.array([2,1,0]),axis=1))
        cflips_tb = np.sum(np.alltrue(np.vstack((s[:-2],s[1:-1],s[2:])).T == np.array([2,1,0]),axis=1))
        cflips_bt = np.sum(np.alltrue(np.vstack((s[:-2],s[1:-1],s[2:])).T == np.array([0,1,2]),axis=1))
        cflipmat_tb.append(cflips_tb)
        cflipmat_bt.append(cflips_bt)
        cflipmat.append(flips)
        count+=1
    print('CHOL Up or down Transition:', sum(cflipmat))
    print('Flip Rate:', sum(cflipmat)/(u.trajectory.totaltime*cholgrp.n_atoms)*1000**4,'inverse s')

    print('T1 - Top to Bottom (2->0)', sum(cflipmat_tb))
    print('T2 - Bottom to Top (0->2)', sum(cflipmat_bt))
    
    if os.path.exists(filename):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not
    highscore = open(filename,append_write)
    highscore.write(str(sum(flipmat)) + " " + str(sum(flipmat_tb)) + " " + str(sum(flipmat_bt)) + " " + " " + str(sum(flipmat_st)) + " " + str(sum(flipmat_ts)) + " " +" " + str(sum(flipmat_bs)) + " " + str(sum(flipmat_sb)) + " " + str(sum(flipmat)/(u.trajectory.totaltime*atomgrp.n_atoms)*1000**4) + " " + str(sum(cflipmat)) + " " + str(sum(cflipmat_tb)) + " " + str(sum(cflipmat_bt)) + " " + str(sum(cflipmat)/(u.trajectory.totaltime*cholgrp.n_atoms)*1000**4) + "\n")
    highscore.close()
  
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
