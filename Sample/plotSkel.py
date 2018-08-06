import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


import numpy as np
import matplotlib.pyplot as plt

from numpy import genfromtxt
import sys, getopt, os
import pdb

def main(argv):

	# csvfile = argv[0];
	csvfile = "00001.csv"

	if not os.path.isfile(csvfile):
		print 'ERROR! ', csvfile,' is not a valid file location'
	        sys.exit()

	try:
		pose3D = genfromtxt(csvfile, delimiter=',')

	
		buff_large = np.zeros((32, 3));
		buff_large[(0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27), :] = pose3D;

		pose3D = buff_large.transpose();

	
		kin = np.array([[0, 12], [12, 13], [13, 14], [15, 14], [13, 17], [17, 18], [18, 19], [13, 25], [25, 26], [26, 27], [0, 1], [1, 2], [2, 3], [0, 6], [6, 7], [7, 8]]);
		order = np.array([0,2,1]);

		mpl.rcParams['legend.fontsize'] = 10

		fig = plt.figure()
		ax = fig.gca(projection='3d')
		ax.view_init(azim=-90, elev=15)

		for link in kin:
			 ax.plot(pose3D[0, link], pose3D[2, link], -pose3D[1, link], linewidth=5.0);

		ax.legend()
	
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		ax.set_aspect('equal')

		X = pose3D[0, :]
		Y = pose3D[2, :]
		Z = -pose3D[1, :]
		max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

		mid_x = (X.max()+X.min()) * 0.5
		mid_y = (Y.max()+Y.min()) * 0.5
		mid_z = (Z.max()+Z.min()) * 0.5
		ax.set_xlim(mid_x - max_range, mid_x + max_range)
		ax.set_ylim(mid_y - max_range, mid_y + max_range)
		ax.set_zlim(mid_z - max_range, mid_z + max_range)
		print("success")
		plt.show()

	except Exception as err:
		print type(err)
		print err.args
		print err
		sys.exit(2)

        

if __name__ == "__main__":
   main(sys.argv[1:])
