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

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import cv2
import glob
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

GENERATE_IMAGES = False

GEN_CAMERA_CALIBRATION_IMG = GENERATE_IMAGES
GEN_DISTORTION_CORRECTION_IMG = GENERATE_IMAGES
GEN_THRESHOLD_COMBINATION_IMG = GENERATE_IMAGES
GEN_PERSPECTIVE_TRANSFORM_IMG = GENERATE_IMAGES
GEN_POLYFIT_IMG = GENERATE_IMAGES
GEN_WARPED_BACK_IMG = GENERATE_IMAGES
GEN_VISUAL_DISPLAY_IMG = GENERATE_IMAGES


chessboard_width = 9  # number of corners in x direction
chessboard_height = 6  # number of corners in y direction
cali_image_path = 'camera_cal/'
distorted_test_img = 'camera_cal/calibration1.jpg'
test_images_path = 'test_images/'
output_images_path = 'output_images/'

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboard_height*chessboard_width, 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_width, 0:chessboard_height].T.reshape(-1, 2)

# https://safety.fhwa.dot.gov/geometric/pubs/mitigationstrategies/chapter3/3_lanewidth.cfm
# use freeway lanewidth 3.6 from above link
# each image is img.shape[0]: 720, img.shape[1]: 1280
xm_per_pix = 3.6 / 720  # 720 pixels in x direction for 3.6 meter
ym_per_pix = 30 / 1280  # 1280 pixels in y direction for 30 meter approxmately

# threshold min and max used in thresholded_binary_img -- and this requires lots tunning!!!
xsobel_min = 30
xsobel_max = 200
magnitude_min = 30
magnitude_max = 200
direction_min = 0
direction_max = np.pi/2
hls_min = 160
hls_max = 255

# For perspective transformation -- we obtained these numbers from "test.jpg" and apply them to other images
# as long as src and dst are matching
# define src corners (from)
top_left = [560, 470]
top_right = [730, 470]
bottom_right = [1080, 720]
bottom_left = [200, 720]
# define transformed dst corners (to)
top_left_dst = [200, 0]
top_right_dst = [1100, 0]
bottom_right_dst = [1100, 720]
bottom_left_dst = [200, 720]

# this is simply for saving the file with correct name
current_img_file = ""


def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """
    Takes an image, gradient orientation, and threshold min/max values
    """
    # convert to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply x or y gradient with the OpenCV Sobel() function and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    elif orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    else:
        print("please specify the direction")
        sys.exit()

    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary


def color_thresh(img, s_thresh, l_thresh, b_thresh, v_thresh):
    """
    Takes an image, apply color threshold min/max values
    """
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    #hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    # extract s channel from hsv,  b from lab， l from luv and v from hsv
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    b_channel = lab[:, :, 2]
    l_channel = luv[:, :, 0]
    # extract pixels from s channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # extract pixels from v channel
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel > v_thresh[0]) & (v_channel <= v_thresh[1])] = 1
    # extract pixels from b channel
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel > b_thresh[0]) & (b_channel <= b_thresh[1])] = 1
    # extract pixels from l channel
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel > l_thresh[0]) & (l_channel <= l_thresh[1])] = 1

    # combine above pixels together
    combined = np.zeros_like(s_channel)
    combined[((s_binary == 1) & (b_binary == 1) & (l_binary == 1) & (v_binary == 1))] = 1

    return combined, s_channel


def rgb_select(img, r_thresh, g_thresh, b_thresh):
    """
    Takes an image, apply RGB color threshold min/max values
    """
    r_channel = img[:, :, 0]
    g_channel = img[:, :, 1]
    b_channel = img[:, :, 2]

    # extract red
    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel > r_thresh[0]) & (r_channel <= r_thresh[1])] = 1

    # extract green
    g_binary = np.zeros_like(g_channel)
    g_binary[(g_channel > g_thresh[0]) & (g_channel <= g_thresh[1])] = 1

    # extract blue
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel > b_thresh[0]) & (b_channel <= b_thresh[1])] = 1

    # combine above pixels together
    combined = np.zeros_like(r_channel)
    combined[((r_binary == 1) & (g_binary == 1) & (b_binary == 1))] = 1

    return combined


def mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100)):
    """
    Return the magnitude of the gradient
    for a given sobel kernel size and threshold values
    """
    # Convert to grayscale
    # hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # s_channel = hls[:, :, 2]
    # tmp = s_channel

    tmp = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(tmp, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(tmp, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    gradmag = np.uint8(255*gradmag/np.max(gradmag))
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Return the direction of the gradient
    for a given sobel kernel size and threshold values
    """
    # Convert to grayscale
    tmp = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Calculate the x and y gradients
    sobelx = cv2.Sobel(tmp, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(tmp, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_output


def measure_curvature_pixels(ploty, leftx, lefty, rightx, righty):
    """
    Calculates the curvature of polynomial functions in pixels.
    """

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


def display_curv_offset(color_warp, undist, left_curve, right_curve, vehicle_offset, m_inv):
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # plt.imshow(result)
    # plt.show()

    # Annotate lane curvature values and vehicle offset from center
    avg_curve = (left_curve + right_curve)/2
    label_str = 'Radius of Curvature = %.1f (m)' % avg_curve
    result = cv2.putText(result, label_str, (30, 40), 0, 1.8, (255, 255, 255), 2, cv2.LINE_AA)

    if vehicle_offset > 0:
        label_str = 'Vehicle is %.1fm right of center' % vehicle_offset
    elif vehicle_offset < 0:
        label_str = 'Vehicle is %.1fm left of center' % -vehicle_offset
    else:
        label_str = 'Vehicle is at center'

    result = cv2.putText(result, label_str, (30, 90), 0, 1.8, (255, 255, 255), 2, cv2.LINE_AA)

    if GEN_VISUAL_DISPLAY_IMG:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(undist)
        ax1.set_title('Undistored', fontsize=25)
        ax2.imshow(result)
        ax2.set_title('Final Result', fontsize=25)
        # plt.show()
        fig.savefig(output_images_path + 'display_with_curv_offset_' + current_img_file + '.png')

    return result


def overlay_detected_lane(undist, left_fit, right_fit):
    """
     Warp the detected lane boundaries back onto the original image.
    """

    # Generate x and y values for plotting
    ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    #warp_zero = np.zeros_like(warped).astype(np.uint8)
    #color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    color_warp = np.zeros((undist.shape[0], undist.shape[1], 3), dtype='uint8')

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    if GEN_WARPED_BACK_IMG:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(undist)
        ax1.set_title('Undistored', fontsize=25)
        ax2.imshow(color_warp)
        ax2.set_title('Polyfilled Lane overlayed', fontsize=25)
        # plt.show()
        fig.savefig(output_images_path + 'overlay_detected_lane_' + current_img_file + '.png')

    return color_warp


def find_lane_pixels(binary_warped):
    """
    extract lane pixels
    """
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
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
    window_height = np.int(binary_warped.shape[0] // nwindows)
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
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
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
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    if GEN_POLYFIT_IMG:
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
        #plt.imshow(out_img)
        #plt.show()
        plt.savefig(output_images_path + 'fit_polynomial_' + current_img_file + '.png')

    return ploty, leftx, lefty, rightx, righty, left_fit, right_fit


def perspective_transform(undistorted_img, combined_img):
    """
    carry perspective transform to map from image to the warped space
    """

    # assemble src and dst
    src = np.float32([top_left, top_right, bottom_right, bottom_left])
    dst = np.float32([top_left_dst, top_right_dst, bottom_right_dst, bottom_left_dst])

    img_size = (undistorted_img.shape[1], undistorted_img.shape[0])

    # get mapping matrix
    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)

    # any difference between INTER_LINEAR and INTER_NEAREST
    warped = cv2.warpPerspective(combined_img, m, img_size)

    if GEN_PERSPECTIVE_TRANSFORM_IMG:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
        ax1.imshow(undistorted_img)
        ax1.set_title('Undistorted', fontsize=25)
        ax2.imshow(combined_img)
        ax2.set_title('Thresholded', fontsize=25)
        ax3.imshow(warped)
        ax3.set_title('Warped', fontsize=25)
        # plt.show()
        fig.savefig(output_images_path + 'perspective_transform_' + current_img_file + '.png')

    return warped, m_inv


def thresholded_binary_img(undistorted_image):
    """
    carry color and gradient transform to get binary image
    """
    kernel_size = 15

    # gradient threshold
    gradx = abs_sobel_thresh(undistorted_image, orient='x', sobel_kernel=kernel_size, thresh=(50, 90))
    grady = abs_sobel_thresh(undistorted_image, orient='y', sobel_kernel=kernel_size, thresh=(30, 90))

    # color threshold
    c_binary, s_channel = color_thresh(undistorted_image, s_thresh=(70, 100), l_thresh=(60, 255),
                                       b_thresh=(50, 255), v_thresh=(150, 255))
    rgb_binary = rgb_select(undistorted_image, r_thresh=(225, 255), g_thresh=(225, 255), b_thresh=(0, 255))

    # direction threshold
    dir_bin = dir_threshold(undistorted_image, sobel_kernel=kernel_size, thresh=(direction_min, direction_max))

    combined_binary = np.zeros_like(s_channel)
    combined_binary[((gradx == 1) & (grady == 1) & (dir_bin == 1) | (c_binary == 1) | (rgb_binary == 1))] = 255

    if GEN_THRESHOLD_COMBINATION_IMG:
        # Plot the result
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        fig.tight_layout()
        ax1.imshow(undistorted_image)
        ax1.set_title('Undistorted', fontsize=40)
        ax2.imshow(combined_binary)
        ax2.set_title('Thresholded', fontsize=40)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        #plt.show()
        fig.savefig(output_images_path + 'thresholded_' + current_img_file + '.png')

    return combined_binary


def correct_distortion(distorted_img):
    """
    compute the distortion coefficients and do the actual correction here
    """
    img = np.copy(distorted_img)

    # Load the saved camera matrix and distortion coefficients
    dist_pickle = pickle.load(open(cali_image_path + 'wide_dist_pickle.p', "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    if GEN_DISTORTION_CORRECTION_IMG:
        # Visualize undistortion
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original', fontsize=25)
        ax2.imshow(undist)
        ax2.set_title('Undistorted', fontsize=25)
        #plt.show()
        fig.savefig(output_images_path + 'correct_distortion_' + current_img_file + '.png')

    return undist


########################################################################
# Process Image
########################################################################
def process_image(img):
    # Apply a distortion correction to raw image
    # undist = correct_distortion(test_images_path + "/straight_lines1.jpg")
    undist = correct_distortion(img)

    # Use color transforms, gradients, etc., to create a thresholded binary image.
    combined_binary = thresholded_binary_img(undist)
    # combined_binary = thresholded_binary_img_old(undist, color_thresh=(200, 255), xgradiet_thresh=(20, 100))

    # Apply a perspective transform to rectify binary image ("birds-eye view").
    warped, m_inv = perspective_transform(undist, combined_binary)

    # Detect lane pixels and fit to find the lane boundary.
    ploty, leftx, lefty, rightx, righty, left_fit, right_fit = fit_polynomial(warped)
    # ploty, left_fit, right_fit = generate_data()   # to validate the algorithm

    # Determine the curvature of the lane and vehicle position with respect to center.
    left_curverad, right_curverad = measure_curvature_pixels(ploty, leftx, lefty, rightx, righty)
    vehicle_offset = calc_vehicle_offset(undist, left_fit, right_fit)

    # Warp the detected lane boundaries back onto the original image.
    overlay_img = overlay_detected_lane(undist, left_fit, right_fit)

    # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    result_img = display_curv_offset(overlay_img, undist, left_curverad, right_curverad, vehicle_offset, m_inv)

    return result_img


########################################################################
# Process Videos
########################################################################
def process_video(video_input, video_output, process_image_fn):
    clip = VideoFileClip(os.path.join('test_videos', video_input))
    processed = clip.fl_image(process_image_fn)
    processed.write_videofile(os.path.join('test_videos_output', video_output), audio=False)


########################################################################
# Camera Calibration
########################################################################
def camera_calibration(calibration_images_path):
    """
    Populate objpoints and imgpoints from all calibration images. Only execute once
    """
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images (all *.jpg files)
    images = glob.glob(calibration_images_path + "*.jpg")

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
            img = cv2.drawChessboardCorners(img, (chessboard_width, chessboard_height), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
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


def main():

    global current_img_file
    # only run this once !!!
    # Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    # camera_calibration(cali_image_path)

    if GEN_CAMERA_CALIBRATION_IMG:
        test_img = cv2.imread(cali_image_path + "calibration1.jpg")
        current_img_file = "calibration1"
        correct_distortion(test_img)

    # # Make a list of calibration images (all *.jpg files)
    # images = glob.glob(test_images_path + "/*.jpg")
    # # Step through the list and search for chessboard corners
    # for fname in images:
    #     print(fname)
    #     current_img_file = os.path.basename(fname)
    #     current_img_file = os.path.splitext(current_img_file)[0]
    #     img = cv2.imread(fname)
    #     result_img = process_image(img)
    #     #plt.imshow(result_img)
    #     #plt.show()
    # cv2.destroyAllWindows()

    process_video('project_video.mp4', 'project_video_output.mp4', process_image)
    # process_video('challenge_video.mp4', 'challenge_video_output.mp4', process_image)
    # process_video('harder_challenge_video.mp4', 'harder_challenge_video_output.mp4', process_image)


main()
