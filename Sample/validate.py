import numpy as np


import sys, getopt
import os, csv
import pdb
from numpy import genfromtxt

def main(argv):
   predfolder = ''
   try:
      opts, args = getopt.getopt(argv,"hp:")
   except getopt.GetoptError:
      print 'USAGE: validate.py -p <predfolder>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'USAGE: validate.py -p <predfolder>'
         sys.exit()
      elif opt in ("-p"):
         predfolder = arg

   if not predfolder:
	print 'USAGE: validate.py -p <predfolder>'
        sys.exit()

   gtfolder = os.path.abspath(os.path.join(os.path.dirname( predfolder ), 'POSE'))

   print 'Getting results from ', predfolder
   print 'Using validation GT from', gtfolder
   
   if not os.path.isdir(predfolder):
	print 'ERROR! ', predfolder,' is not a valid location'
        sys.exit()
   if not os.path.isdir(gtfolder):
	print 'ERROR! ', gtfolder,' is not a valid location'
        sys.exit()

   
   error = np.zeros((19312,))
   for i in range(0,19312):
	try:
		gtfile = os.path.join(gtfolder, '%05d.csv' % (i+1))
		gtPose = genfromtxt(gtfile, delimiter=',')
		predfile = os.path.join(predfolder, '%05d.csv' % (i+1))
		predPose = genfromtxt(predfile, delimiter=',')
		error[i] = np.mean(np.sqrt(np.sum((gtPose - predPose) ** 2, axis = 1)))
	except Exception as err:
		print type(err)
		print err.args
		print err
		sys.exit(2)

   print '=========> THE ERROR FOR THE VALIDATION SET IS %.3f MM' % np.mean(error)

if __name__ == "__main__":
   main(sys.argv[1:])
