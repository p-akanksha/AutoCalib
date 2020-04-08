import os
import cv2
import math
import numpy as np

def compute_v(h1, h2):
	v = np.zeros((1,6))

	v[0][0] = h1[0]*h2[0]
	v[0][1] = h1[0]*h2[1] + h1[1]*h2[0]
	v[0][2] = h1[1]*h2[1]
	v[0][3] = h1[2]*h2[0] + h1[0]*h2[2]
	v[0][4] = h1[2]*h2[1] + h1[1]*h2[2]
	v[0][5] = h1[2]*h2[2]

	return v

def find_K(homographies):
	n, _, _ = homographies.shape
	
	# calculate v
	V = np.zeros((2*n, 6))
	for i, H in enumerate(homographies):
		h1 = H[:, 0]
		h2 = H[:, 1]
		h3 = H[:, 2]

		v11 = compute_v(h1, h1)
		v22 = compute_v(h2, h2)
		v12 = compute_v(h1, h2)

		# v22 = compute_v(h2, h2)
		# v33 = compute_v(h3, h3)
		# v23 = compute_v(h2, h3)

		V[2*i] = v12
		V[2*i + 1] = v11 - v22

	# calculate b
	# print(V)
	U, S, Vs = np.linalg.svd(V)
	# z = np.zeros((26,6))
	# z[0][0] = S[0]
	# z[1][1] = S[1]
	# z[2][2] = S[2]
	# z[3][3] = S[3]
	# z[4][4] = S[4]
	# z[5][5] = S[5]
	# print(U.shape)
	# print(S.shape)
	# print(Vs.shape)
	b = Vs[-1, :]
	# x = np.matmul(z,Vs)
	# y = np.matmul(U,x)
	# print(y)

	# get patameters of K
	v0 = (b[1]*b[3] - b[0]*b[4])/(b[0]*b[2] - b[1]**2)
	lam = b[5] - (b[3]**2 + v0*(b[1]*b[3] - b[0]*b[4]))/b[0]
	alpha = math.sqrt(lam/b[0])
	# print((b[0]*b[2] - b[1]**2))
	beta = math.sqrt((lam*b[0])/(b[0]*b[2] - b[1]**2))
	gamma = -(b[1]*(alpha**2)*beta)/lam
	u0 = gamma*v0/beta - b[3]*(alpha**2)/gamma

	K = np.zeros((3,3))
	K[0][0] = alpha
	K[0][1] = gamma
	K[0][2] = u0
	K[1][1] = beta
	K[1][2] = v0
	K[2][2] = 1 

	return K


def estimate_K(images):
	# coordinates of four inner corners in world frame
	points = np.asarray([[21.5, 21.5],
						 [193.5, 21.5],
			   			 [21.5, 129], 
			   			 [193.5, 129]])

	homographies = []

	for im in images:
		# get coordinates of corners
		rv, cor = cv2.findChessboardCorners(im, (9,6))

		if not rv:
			print("Cannot detect chessboard")
			continue

		# corners corresponding to selected points 
		corners = np.asarray([[cor[0][0][0], cor[0][0][1]],
				  		  	  [cor[8][0][0], cor[8][0][1]],
				  			  [cor[45][0][0], cor[45][0][1]],
				  			  [cor[53][0][0], cor[53][0][1]]])

		# calculate homography
		H, mask = cv2.findHomography(points, corners)
		homographies.append(H)

	homographies = np.asarray(homographies)
	# print(homographies.shape)

	K = find_K(homographies)

	return K


def main():
	cur_dir = os.path.dirname(os.path.abspath(__file__))
	img_path = os.path.join(cur_dir, "Calibration_Imgs")

	# read images
	images = []
	for name in sorted(os.listdir(img_path)):
		im = cv2.imread(os.path.join(img_path, name))
		images.append(im)
	images = np.asarray(images)

	# get initial estimate of K
	K_in = estimate_K(images)


if __name__ == '__main__':
	main()