#!/usr/bin/python

from optparse import OptionParser
import os
import sys
import shutil
import subprocess


# user settings 
g_prefix='gmx '

parser = OptionParser()
parser.add_option("-w", "--wdir",
                      action="store",
                      metavar=" ",
                      dest="wdir",
                      default="work",
                      help="Working Directory of project (default:work)")
parser.add_option("-v","--verbose",
                       action="store_true",
                       dest="verbose",
                       default=True,
                       help="Loud and Noisy[default]")
parser.add_option("-q","--quiet", 
                      action="store_false",
                      dest="verbose",
                      help="Be vewwy quit")
parser.add_option("-s", "--tpr",
                      action="store",
                      metavar=" ",
                      dest="tpr",
                      default="large-ld-md1.tpr",
                      help="Input a finished run input file [*.tpr]")
parser.add_option("-x", "--xtc",
                      action="store",
                      metavar=" ",
                      dest="traj",
                      default="large-ld-md1.xtc",
                      help="Input a trajectory file [*.xtc]")
parser.add_option("-g", "--gro",
                      action="store",
                      metavar=" ",
                      dest="gro",
                      default="large-ld-md1.gro",
                      help="Input a gro file [*.gro]")
parser.add_option("-m", "--mdcount",
                      action="store",
                      metavar=" ",
                      dest="mdcount",
                      default="md1",
                      help="Give which trail run we want to use")
parser.add_option("-a", "--alcname",
                      action="store",
                      metavar=" ",
                      dest="alcname",
                      default="oct",
                      help="Give alcohol label to output")
parser.add_option("-p", "--phasename",
                      action="store",
                      metavar=" ",
                      dest="pha",
                      default="ld",
                      help="Give phase label to output")

(options, args) = parser.parse_args()
print("args "+ str(len(args)))
if len(args) == 4:
    parser.error("wrong number of arguments")
    
global otpr, otraj
otpr=options.tpr
otraj=options.traj
ogro=options.gro
md=options.mdcount
alc=options.alcname
phase=options.pha
wdir=options.wdir+"/"
ntpr="n"+alc+"-"+phase+"-"+md+ ".tpr"
PO4xvg=alc+"-"+phase+"-"+"PO4"+md+ ".xvg"
ROHxvg=alc+"-"+phase+"-"+"ROH"+md+ ".xvg"
alcxvg=alc+"-"+phase+"-"+md+ ".xvg"
ntraj=alc+"-"+phase+"-"+md+ ".xtc"
ngro="n"+alc+"-"+phase+"-"+md+ ".gro"
ntraj="n"+alc+"-"+phase+"-"+md+ ".xtc"
nndx="n"+alc+"-"+phase+"-"+md+ ".ndx"

def indexer():
    print(">STEP1 : Initiating Procedure to generate index")
    StepNo = "1"
    StepName = "Compact Index"
    p = subprocess.Popen(['gmx', 'make_ndx',
                      '-f', otpr, '-o', nndx],
                     stdin=subprocess.PIPE)
    p.communicate(b'2|3\n7&aPO4\n4&aROH\n8|9|5\nq\n') # Center on protein, output everything
    p.wait()

def shortfiles():
    print(">STEP2 : Initiating Procedure to generate New XTC, GRO, and TPR")
    StepNo = "2"
    StepName = "XTC generation"
    p = subprocess.Popen(['gmx', 'trjconv',
                      '-f',otraj, '-n', nndx,
                      '-s',otpr,'-o',ntraj],
                     stdin=subprocess.PIPE)
    p.communicate(b'10\nq\n') # Center on membrane, calculate head density
    p.wait()
    StepName = "GRO generation"
    p = subprocess.Popen(['gmx', 'trjconv',
                      '-f',otraj, '-n', nndx,
                      '-s',otpr,'-o',ngro, '-b','3000000','-e','3000001'],
                     stdin=subprocess.PIPE)
    p.communicate(b'10\nq\n') # Center on membrane, calculate head density
    p.wait()
    StepName = "TPR generation"
    p = subprocess.Popen(['gmx', 'convert-tpr',
                      '-s',otpr, '-n', nndx,
                      '-o',ntpr],
                     stdin=subprocess.PIPE)
    p.communicate(b'10\nq\n') # Center on membrane, calculate head density
    p.wait()
    StepName = "Compact Index 2"
    p = subprocess.Popen(['gmx', 'make_ndx',
                      '-f', ntpr, '-o', nndx],
                     stdin=subprocess.PIPE)
    p.communicate(b'2|3\n2|3|4\nq\n') # Center on protein, output everything
    p.wait()

def dens():
    print(">STEP3 : Initiating Procedure to generate Desnity for lipids")
    StepNo = "3"
    StepName = "Density Generation for PO4 Heads"
    p = subprocess.Popen(['gmx', 'density',
                      '-nocopyright','-sl','200','-s', ntpr,'-f'
                      ,ntraj, '-n', nndx,'-center',
                      '-d','Z','-dens','number','-o',PO4xvg],
                     stdin=subprocess.PIPE)
    p.communicate(b'7\n6\nq\n') # Center on membrane, calculate head density
    p.wait()
    StepName = "Density Generation for CHOL heads"
    p = subprocess.Popen(['gmx', 'density',
                      '-nocopyright','-sl','200','-s', ntpr,'-f'
                      ,ntraj, '-n', nndx,'-center',
                      '-d','Z','-dens','number','-o',ROHxvg],
                     stdin=subprocess.PIPE)
    p.communicate(b'7\n4\nq\n') # Center on membrane, calculate ROH density
    p.wait()
    StepName = "Density Generation for ALC heads"
    p = subprocess.Popen(['gmx', 'density',
                      '-nocopyright','-sl','200','-s', ntpr,'-f'
                      ,ntraj, '-n', nndx,'-center',
                      '-d','Z','-dens','number','-o',alcxvg],
                     stdin=subprocess.PIPE)
    p.communicate(b'7\n5\nq\n') # Center on membrane, calculate ROH density
    p.wait()    

if __name__ == '__main__':
    #GatherFiles()
    indexer()
    shortfiles()
    dens()
    print("""
    *****************************************************
                   COMPLETED SUCCESSFULLY 
    *****************************************************
""")
