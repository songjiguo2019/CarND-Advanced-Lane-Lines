#Advanced Lane Finding Project
#
#The goals / steps of this project are the following:
#
#    Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
#    Apply a distortion correction to raw images.
#    Use color transforms, gradients, etc., to create a thresholded binary image.   +
#    Apply a perspective transform to rectify binary image ("birds-eye view").
#    Detect lane pixels and fit to find the lane boundary.
#    Determine the curvature of the lane and vehicle position with respect to center.
#    Warp the detected lane boundaries back onto the original image.
#    Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
#

#First, I'll compute the camera calibration using chessboard images

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import cv2
import glob

import sys

chessboard_width = 9  # number of corners in x direction
chessboard_height = 6  # number of corners in y direction
cali_image_path = '../camera_cal/'
distorted_test_img = '../camera_cal/calibration1.jpg'
test_images_path = '../test_images/'
output_images_path = '../output_images/'

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboard_height*chessboard_width, 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_width, 0:chessboard_height].T.reshape(-1, 2)

#ym_per_pix = 30/720 # meters per pixel in y dimension
ym_per_pix = 3.7 / 700  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension


# this function is only used for testing purpose
def generate_data():
    """
    Generates fake data to use for calculating lane curvature.
    In your own project, you'll ignore this function and instead
    feed in the output of your lane detection algorithm to
    the lane curvature calculation.
    """
    # Set random seed number so results are consistent for grader
    # Comment this out if you'd like to see results on different random data!
    np.random.seed(0)
    # Generate some fake data to represent lane-line pixels
    ploty = np.linspace(0, 719, num=720)  # to cover same y-range as image
    quadratic_coeff = 3e-4  # arbitrary quadratic coefficient
    # For each y position generate random x position within +/-50 pix
    # of the line base position in each case (x=200 for left, and x=900 for right)
    leftx = np.array([200 + (y ** 2) * quadratic_coeff + np.random.randint(-50, high=51)
                      for y in ploty])
    rightx = np.array([900 + (y ** 2) * quadratic_coeff + np.random.randint(-50, high=51)
                       for y in ploty])

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit = np.polyfit(ploty, leftx, 2)
    right_fit = np.polyfit(ploty, rightx, 2)

    return ploty, left_fit, right_fit


# carry color and gradient transform to get binary image
def thresholded_binary_img(img, color_thresh=(200, 255), xgradiet_thresh=(20, 100)):
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= xgradiet_thresh[0]) & (scaled_sobel <= xgradiet_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= color_thresh[0]) & (s_channel <= color_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()    
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=40)    
    ax2.imshow(combined_binary)
    ax2.set_title('Thresholded Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

    return combined_binary


# compute the distortion coefficients and do the actual correction here
def correct_distortion(distorted_img):        
    img = cv2.imread(distorted_img)

    # Load the saved camera matrix and distortion coefficients
    dist_pickle = pickle.load(open(cali_image_path + 'wide_dist_pickle.p', "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    cv2.imwrite(output_images_path + 'test_undist.jpg', undist)

    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(undist)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.show()

    return undist


# this function will populate objpoints and imgpoints from all calibration images
# this function should only execute once and we save all distortion coefficient
def camera_calibration(calibration_images_path):
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images (all *.jpg files)
    images = glob.glob(calibration_images_path + "/*.jpg")

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (chessboard_width, chessboard_height), None)
    
        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
    
            # Draw and display the corners
            #img = cv2.drawChessboardCorners(img, (chessboard_width, chessboard_height), corners, ret)
            #cv2.imshow('img',img)
            #cv2.waitKey(500)
        else:
            print('Can not find corners: ret = %s for %s' % (ret, fname))

    # get image info from any one of test images
    img = cv2.imread(distorted_test_img)
    img_size = (img.shape[1], img.shape[0])
    
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                       img_size, None, None)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = dict()
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open(cali_image_path + 'wide_dist_pickle.p', "wb"))


def perspective_transform(img):
    # Note!!!! -- we obtained these numbers from "straight_lines1.jpg" and apply them to other images
    # Note!!!! -- we need get those x,y from correct image!
    src = np.float32([[330, 690], [1090, 690], [573, 460], [700, 460]])
    dst = np.float32([[300, 720], [980, 720],  [300, 0],   [980, 0]])

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    
    # any difference between INTER_LINEAR and INTER_NEAREST
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)  # keep same size as input image
    unwarped = cv2.warpPerspective(warped, M_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR) # DEBUG

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(warped)
    ax2.set_title('Warped Image', fontsize=30)
    ax3.imshow(unwarped)
    ax3.set_title('UnWarped Image', fontsize=30)
    plt.show()

    return warped, M_inv


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    plt.imshow(out_img)
    plt.show()
    
    return ploty, leftx, lefty, rightx, righty, left_fit, right_fit


def measure_curvature_pixels(ploty, leftx, lefty, rightx, righty):
    """
    Calculates the curvature of polynomial functions in pixels.
    """

    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    #ploty, left_fit, right_fit = generate_data()
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix +
                           left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix +
                            right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
    # Now our radius of curvature is in meters

    return left_curverad, right_curverad


def calc_vehicle_offset(undist, left_fit, right_fit):
    """
    Calculate vehicle offset from lane center, in meters or in pixels
    """

    bottom_y = undist.shape[0] - 1
    bottom_x_left = left_fit[0]*(bottom_y**2) + left_fit[1]*bottom_y + left_fit[2]
    bottom_x_right = right_fit[0]*(bottom_y**2) + right_fit[1]*bottom_y + right_fit[2]
    vehicle_offset = undist.shape[1]/2 - (bottom_x_left + bottom_x_right)/2

    # Convert pixel offset to meters
    vehicle_offset *= xm_per_pix

    return vehicle_offset


def final_viz(undist, left_fit, right_fit, m_inv, left_curve, right_curve, vehicle_offset):
    """
    Final lane line prediction visualized and overlayed on top of original image
    """

    # Generate x and y values for plotting
    ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    #warp_zero = np.zeros_like(warped).astype(np.uint8)
    #color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    color_warp = np.zeros((720, 1280, 3), dtype='uint8')  # NOTE: Hard-coded image dimensions

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # Annotate lane curvature values and vehicle offset from center
    avg_curve = (left_curve + right_curve)/2
    label_str = 'Radius of curvature: %.1f m' % avg_curve
    result = cv2.putText(result, label_str, (30,40), 0, 1, (0,0,0), 2, cv2.LINE_AA)

    label_str = 'Vehicle offset from lane center: %.1f m' % vehicle_offset
    result = cv2.putText(result, label_str, (30,70), 0, 1, (0,0,0), 2, cv2.LINE_AA)

    plt.imshow(result)
    plt.show()

    return result


def main():

    # only run this once !!!
    # Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    # camera_calibration(cali_image_path)

    # Make a list of calibration images (all *.jpg files)
    images = glob.glob(test_images_path + "/*.jpg")

    # Step through the list and search for chessboard corners
    for fname in images:

        print(fname)

        # Apply a distortion correction to raw image
        # undist = correct_distortion(test_images_path + "/straight_lines1.jpg")
        undist = correct_distortion(fname)

        # Use color transforms, gradients, etc., to create a thresholded binary image.
        combined_binary = thresholded_binary_img(undist, color_thresh=(200, 255), xgradiet_thresh=(20, 100))

        # Apply a perspective transform to rectify binary image ("birds-eye view").
        warped, M_inv = perspective_transform(combined_binary)

        # Detect lane pixels and fit to find the lane boundary.
        ploty, leftx, lefty, rightx, righty, left_fit, right_fit = fit_polynomial(warped)
        # ploty, left_fit, right_fit = generate_data()   # to validate the algorithm

        # Determine the curvature of the lane and vehicle position with respect to center.
        left_curverad, right_curverad = measure_curvature_pixels(ploty, leftx, lefty, rightx, righty)
        print(left_curverad, right_curverad)

        vehicle_offset = calc_vehicle_offset(undist, left_fit, right_fit)
        print(vehicle_offset)

        # Warp the detected lane boundaries back onto the original image.

        final_viz(undist, left_fit, right_fit, M_inv, left_curverad, right_curverad, vehicle_offset)

        # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

    cv2.destroyAllWindows()


main()
