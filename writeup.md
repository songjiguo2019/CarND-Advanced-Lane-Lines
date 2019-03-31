## Advanced Lane Finding Project

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[pipeline]: ./pipeline.png "Pipeline"
[image1]: ./output_images/correct_distortion_calibration1.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image2_result]: ./output_images/correct_distortion_test1.png "Road Transformed"
[image3_result]: ./output_images/thresholded_test1.png "Binary Example"
[image4_result]: ./output_images/perspective_transform_test1.png "Warp Example"
[image5_result]: ./output_images/fit_polynomial_test1.png "Fit Visual"
[image6_result]: ./output_images/display_with_curv_offset_test1.png "Output"
[video1]: ./test_videos_output/project_video_output.mp4 "Video"

[//]: # (To remove png margin, "convert input.png -trim output.png")

### Writeup / README

#### All code can be found in **P2.py**
#### Here is an picture of high-level pipeline and how I addressed each one.  

![alt text][pipeline]

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained as function "_camera_calibration()_" in lines #578 through # of the file called `P2.py`.  

The function iterates all chessboard images under "/camera_cal" to prepare "object points", which will be the (x, y, z) 
coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at 
z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of 
coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners 
in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane 
with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using 
the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image (_calibration1.jpg_) using 
the `cv2.undistort()` function. Note: the camera calibration **only executes once** and I have saved computed camera matrix (mtx) and distortion 
coefficients (dist) into a **pickle file** called _wide_dist_pickle.p_ under "/camera_cal", which will be loaded in the 
following pipeline.

### Pipeline (with single images)

#### 1. Provide an example of a distortion-corrected image.

Using above mentioned camera calibration matrices (mtx) and distortion coefficients (dist) in '_wide_dist_pickle.p_', 
I undistorted the input image. To demonstrate this step, I will describe how I apply the distortion correction to one 
of the test images like this one:
![alt text][image2]

After applying _correct_distortion_ function onto above image, we show the undistorted version with orignal one side by 
side as following:

![alt text][image2_result]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

To more clearly and reliably extract lane pixels, I apply _thresholded_binary_img()_ function onto previous undistorted
image to obtain a thresholded binary image. In the function, I used a combination of color and gradient thresholds to 
generate the binary image (thresholding steps at lines #476 through #491 in `P2.py`) as followings: 

```python
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
```

 * Sobel operator in both horizontal and vertical directions
 * Convert the image from RGB space to HSV space, and threshold the S channel
 * Convert the image from RGB space to HSV space, and threshold the V channel
 * Convert the image from RGB space to LAB space, and threshold the B channel
 * Convert the image from RGB space to LUV space, and threshold the L channel
 * Extract Blue channel from RGB space
 * Sobel operator to calculate the direction of the gradient
 * Combine the above binary images to create the final binary image
 
Notice that after many tries, I have to use multiple different color space and corresponding channel to make the final 
result stable, particularly those regions under the shadow. Here is the output thresholed binary image that combined 
the above thresholded binary filters and displayed side-bys-side to undistorted image:

![alt text][image3_result]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform()`, which appears at lines #442 
in `P2.py`. The `perspective_transform()` function takes as inputs previous thresholded image (`combined_binary`), 
and then use some hardcoded source (`src`) and destination (`dst`) points as following (see at the top of `P2.py`):

```python
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
```
I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test 
image, its thresholded version and warped counterpart side-by-side to verify that the lines appear correctly in 
the warped image.

![alt text][image4_result]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I used histogram method to identify left lane and right lane respectively. We can then locate the position of center
of two lanes in x direction (see _find_lane_pixels()_ function at line #324). From here, we apply a 2nd order polynomial 
with sliding windows to fit my lane lines (at line #407 in function _fit_polynomial()_ in file `P2.py`): 

![alt text][image5_result]

As you can see, we show 3 images (undistorted, thresholded and warped) side by side, where warped one now has lines be
drawn on left lane and right lane correctly. 

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the radius of curvature of the lane in function _measure_curvature_pixels()_ at line #216 by fitting new 
polynomials to x,y in world space. For the meter in x and y direction, I hardcoded them according to the image (and also 
https://safety.fhwa.dot.gov/geometric/pubs/mitigationstrategies/chapter3/3_lanewidth.cfm which defines the standard 
lane width in x direction). And I computed vehicle offset by simply getting the offset between the center of lanes and 
the center of the image in x direction in function _calc_vehicle_offset_ at line #239 in my code in `P2.py`.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step at lines #254 my code in `P2.py` in the function `display_curv_offset()`.  Here is an example 
of my result on a test image (side by side to undistorted image):

![alt text][image6_result]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./test_videos_output/project_video_output.mp4)

You can find the video under "test_videos_output/". Indeed the resulted video looks prery stable even when the vehicle is driving by the area that has both shadow and curve :-)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Through the project, I learned and applied some camera calibration and image distortion correction skill. Also I 
learned how to filter the image through color space and warp perspective transformation. Some very basic geometric and 
curve fitting techniques help me get the radius of curvature and also finding the lanes. 

Though this project is obviously more capable in handling shadow and curve compared to the previous project (P1), it 
definitely can be improved to handle more complex environment:

* `challenge_video.mp4` video includes roads with cracks which could be mistaken as lane lines
* `harder_challenge_video.mp4` video includes roads with some turns that have very small radius of curvature, along with 
shadow changes.
* Other vehicles in front (driving in the same direction or coming from opposite direction) would make the lane finder 
algorithm think the vehicle was part of the lane.
* All videos have assumed day time driving. And it might become more challenging during the night or different weather.
Different ambient light condition might incur more challenge to my lane finding algorithm.
* Expect to learn more skills (i.e., some machine learning technique) in the next project to make the lane detector 
more robust. 
* Finally, I think the dramtic change of shadow and small curves are biggest challenges in this project (I spend most 
time on tunning threshold parameters). 

