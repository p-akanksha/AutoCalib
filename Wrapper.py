import os
import cv2
import math
import numpy as np
from scipy.optimize import least_squares

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

		V[2*i] = v12
		V[2*i + 1] = v11 - v22

	# calculate b
	U, S, Vh = np.linalg.svd(V)
	b = Vh[-1, :]

	# get patameters of K
	v0 = (b[1]*b[3] - b[0]*b[4])/(b[0]*b[2] - b[1]**2)
	lam = b[5] - (b[3]**2 + v0*(b[1]*b[3] - b[0]*b[4]))/b[0]
	alpha = math.sqrt(lam/b[0])
	beta = math.sqrt((lam*b[0])/(b[0]*b[2] - b[1]**2))
	gamma = -(b[1]*(alpha**2)*beta)/lam
	u0 = gamma*v0/beta - b[3]*(alpha**2)/gamma

	# construct K
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
	corners = []

	for im in images:
		# get coordinates of corners
		rv, cor = cv2.findChessboardCorners(im, (9,6))

		if not rv:
			print("Cannot detect chessboard")
			continue

		# corners corresponding to selected points 
		c = np.asarray([[cor[0][0][0], cor[0][0][1]],
				  		[cor[8][0][0], cor[8][0][1]],
				  		[cor[45][0][0], cor[45][0][1]],
				  		[cor[53][0][0], cor[53][0][1]]])
		corners.append(cor)

		# calculate homography
		H, mask = cv2.findHomography(points, c)
		homographies.append(H)

	homographies = np.asarray(homographies)
	corners = np.asarray(corners)

	K = find_K(homographies)

	return K, corners, homographies

def init_est(K):
	''' consolidate initial estimates into correct format 
	to pass it as an input to the optimizer
	
	First 5 parameters correspond to values of K (alpha,
	beta, gamma, u0, v0) and the last two correspond to 
	distortion parameters (k1, k2)'''
	est = np.zeros((7,1))

	# (k1, k2) = (0, 0)
	est[0] = K[0][0]
	est[1] = K[1][1]
	est[2] = K[0][1]
	est[3] = K[0][2]
	est[4] = K[1][2]

	return est

def find_Rt(K, H):
	K_inv = np.linalg.inv(K)
	h1 = H[:, 0]
	h2 = H[:, 1]
	h3 = H[:, 2]

	lam = 1/np.linalg.norm(np.matmul(K_inv, h1))

	r1 = lam*np.matmul(K_inv, h1)
	r2 = lam*np.matmul(K_inv, h2)
	r3 = np.cross(r1, r2)
	t = lam*np.matmul(K_inv, h3)
	t = np.reshape(t, (3,1))

	Q = np.array([r1, r2, r3]).transpose()

	U, S, Vh = np.linalg.svd(Q)
	R = np.matmul(U, Vh)

	Rt = np.hstack([R, t])
	return Rt


def rep_error(param, corners, homographies):
	K = np.zeros((3,3))
	D = np.zeros((2,1))

	K[0][0] = param[0]
	K[1][1] = param[1]
	K[0][1] = param[2]
	K[0][2] = param[3]
	K[1][2] = param[4]
	K[2][2] = 1

	D[0] = param[5]
	D[1] = param[6]

	points = []
	for i in range(6):
		for j in range(9):
			points.append([21.5*(j+1),21.5*(i+1),0,1])
	points = np.array(points)
	points = points.T

	error = []
	
	for i, pts in enumerate(corners):
		# estimate Rt
		Rt = find_Rt(K, homographies[i])

		# Projection Matrix
		P = np.matmul(K, Rt)

		# Get points in camera coordinate system
		pt_cam = np.matmul(Rt, points)
		pt_cam = pt_cam/pt_cam[2]

		# Get points in image coordinate system
		pt_img = np.matmul(P, points)
		pt_img = pt_img/pt_img[2]

		# Reprojected points
		u_hat = pt_img[0] + (pt_img[0] - K[0][2])*(D[0]*(pt_cam[0]**2 + pt_cam[1]**2) + D[1]*(pt_cam[0]**2 + pt_cam[1]**2)**2)
		v_hat = pt_img[1] + (pt_img[1] - K[1][2])*(D[0]*(pt_cam[0]**2 + pt_cam[1]**2) + D[1]*(pt_cam[0]**2 + pt_cam[1]**2)**2)

		reproj = np.vstack([u_hat, v_hat])
		reproj = reproj.T

		pts = pts.reshape((54, 2))

		x = np.subtract(reproj, pts)
		e = (np.linalg.norm(np.subtract(reproj, pts), axis = 1))**2
		error.append(e)

	error = np.float32(error)
	error = np.asarray(error) 
	error = error.reshape(702)	

	return error

def RMSerror(K, D, homographies, corner_points):

	points = []
	for i in range(6):
		for j in range(9):
			points.append([21.5*(j+1),21.5*(i+1),0])
	points = np.array(points)

	mean = 0
	error = np.zeros([2,1])
	for i in range(homographies.shape[0]):
		# find extrinsic parameters
		Rt = find_Rt(K, homographies[i])

		# Reprojected points
		img_points, _ = cv2.projectPoints(points, Rt[:,0:3], Rt[:,3], K, D)
		img_points = np.array(img_points)

		# calculate error
		errors = np.linalg.norm(corner_points[i,:,0,:]-img_points[:,0,:],axis=1)
		error = np.concatenate([error,np.reshape(errors,(errors.shape[0],1))])

	mean_error = np.mean(error)
	return mean_error

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
	K, corners, homographies = estimate_K(images)
	np.set_printoptions(formatter={'float_kind':'{:f}'.format})
	print("Initial estimate of K: ")
	print(K)

	# Get data in correct format to pass to optimizer
	est = init_est(K)

	# optimize the values of K
	res = least_squares(rep_error,x0=np.squeeze(est),method='lm',args=(corners, homographies))

	# construct K and D matrix
	K = np.zeros((3,3))
	D = np.zeros((5,1))

	K[0][0] = res.x[0]
	K[1][1] = res.x[1]
	K[0][1] = res.x[2]
	K[0][2] = res.x[3]
	K[1][2] = res.x[4]
	K[2][2] = 1

	D[0] = res.x[5]
	D[1] = res.x[6]

	print("intrinsic parameters:")
	print(K)
	print("distortion coefficients:")
	print(D)

	undist_images = []
	# h, w, _ = images[0].shape
	# dim = (int(0.4*w), int(0.4*h))
	for i, im in enumerate(images):
		un = cv2.undistort(im, K, D)
		cv2.imwrite("UndistortedImages/undistImg" + str(i) + ".png", un)

		# Uncomment to display images 
		# cv2.imshow("image", cv2.resize(im, dim))
		# if cv2.waitKey(0) & 0xff == 27:
		# 	cv2.destroyAllWindows()
		# cv2.imshow("undist", cv2.resize(un, dim))
		# if cv2.waitKey(0) & 0xff == 27:
		# 	cv2.destroyAllWindows()

	e = RMSerror(K, D, homographies, corners)
	print("Reprojection error: ", e)

if __name__ == '__main__':
	main()