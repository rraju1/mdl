# Runs dumped checkpoints

#!/usr/bin/env python
import os
import sys
import shutil
import errno

#config = sys.argv[1]
rundir_val = sys.argv[1]
gem5dir = '/home/rraju2/ece752/gem5'
home_dir = '/home/rraju2/ece752'
#print 'Gokul 0'
from spec2k6_benchmarks import benchmarks
submit_script=""
maindir = '/home/rraju2/ece752'
for benchmark,settings in benchmarks.items():
 # print 'Gokul 1'
  checkpoint_dir = os.path.join(home_dir, "simpoints_std", benchmark, "checkpoint") #TODO
  simpoint_file = os.path.join(home_dir, "simpoints_std", benchmark, "simpoints") #TODO
  if os.path.isfile(simpoint_file) and os.access(simpoint_file,os.R_OK): #Gokul
    print "Works"
  else:
    print "Skip"
    continue
  #simpoints = sum(1 for line in open(simpoint_file))
  simpoints = 1 # TODO
  #simpoints = 1  # TODO only one simpoint (0)
  for simpoint in xrange(simpoints):
    #print 'Gokul 2'
    # gem5 binary
    rundir = os.path.join(home_dir, rundir_val, benchmark, str(simpoint)) #Here
    shutil.copytree(maindir + "/data/" + benchmark, rundir)
    shutil.copy(maindir + "/binaries/" + benchmark, rundir)

    gem5cmd = gem5dir + "/build/ARM/gem5.opt " #TODO
    # simulator python script + config
    #TODO
    gem5cmd += gem5dir + "/configs/example/se.py + " "--cpu-type=O3_ARM_v7a_3 --caches --l2cache --mem-size=2GB --cpu-clock=2GHz --sys-clock=2GHz --maxinsts=1000000000 --fast-forward=1000000000 " #Here
    #Gen BBV
    #gem5cmd += gem5dir + "/configs/example/se.py --simpoint-profile --simpoint-interval 100000000 --cpu-type=atomic --fastmem --mem-size=2GB "
    # application binary
    gem5cmd += "--cmd=" + benchmark + " "
    # application arguments
    if 'options' in settings:
         gem5cmd += '--options=\"' + settings['options'] + '\" '
    # setup input/output
    if 'stdin' in settings:
         gem5cmd += "--input=" + settings['stdin'] + " "

    gem5cmd += "--output=" + benchmark + ".stdout "
    gem5cmd += "--errout=" + benchmark + ".stderr "

    sh_script = """#!/bin/sh
    %s"""  % gem5cmd

    sh_file = open(rundir + "/job.sh", "w+")
    sh_file.write(sh_script)
    sh_file.close()

    print sh_script
    submit_script+="""
executable = /bin/sh
arguments = %s
initialdir = %s
output = %s
error = %s
log = %s
Rank=TARGET.Mips
universe = vanilla
getenv = true
queue
    """ % (rundir + "/job.sh",
           rundir,
           rundir + "/gem5sim.out",
           rundir + "/gem5sim.err",
           rundir + "/gem5sim.log")

print submit_script
sub_file = open(rundir_val + "_condor_submit.scr", "w+") #TODO
#sub_file = open("condor_submit.scr", "w+")
sub_file.write(submit_script)
sub_file.close()
#os.system("condor_submit condor_submit.scr")
os.system("condor_submit " + rundir_val + "_condor_submit.scr")
