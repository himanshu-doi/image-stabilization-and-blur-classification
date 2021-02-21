"""
Script to perform image stabilization using Point feature matching
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from memory_profiler import profile
from accessories.time_profiler import timeProfile

source_path = "/home/himanshu/Downloads/OneDrive_1_2-12-2021"
smoothing_radius = 1


def read_frames(source_path):
    frame_list = []
    for f_path in os.listdir(source_path):
        img = cv2.imread(os.path.join(source_path, f_path))
        frame_list.append(img)
    return frame_list


def get_feature_transforms(prev_frame_bw, frame_list, num_frames):
    transforms = np.zeros((n_frames - 1, 3), np.float16)
    for i in range(1, num_frames - 2):
        # Detect feature points in previous frame
        # Detects features with sharp edges
        prev_pts = cv2.goodFeaturesToTrack(prev_frame_bw,
                                           maxCorners=200,
                                           qualityLevel=0.01,
                                           minDistance=30,
                                           blockSize=3)

        # fetch next frame
        curr = frame_list[i]
        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_frame_bw, curr_gray, prev_pts, None)
        # Sanity check
        assert prev_pts.shape == curr_pts.shape

        # Filter only valid points
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        print(m)

        # Extract translation
        dx = m[0, 2]
        dy = m[1, 2]

        # Extract rotation angle
        da = np.arctan2(m[1, 0], m[0, 0])

        # Store transformation
        transforms[i] = [dx, dy, da]

        # Move to next frame
        prev_frame_bw = curr_gray
        print("Frame: " + str(i) + "/" + str(num_frames) + " -  Tracked points : " + str(len(prev_pts)))
    return transforms


def moving_average(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size)/window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed


def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:, i] = moving_average(trajectory[:, i], radius=smoothing_radius)
    return smoothed_trajectory


def fix_border(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    scaled = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
    frame = cv2.warpAffine(frame, scaled, (s[1], s[0]))
    return frame


@profile
@timeProfile(lines_to_print=30, strip_dirs=True)
def get_stabilized_frames():
    # Pre-define transformation-store array
    transforms = get_feature_transforms(prev_gray, frames_list, n_frames)

    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)
    print("trajectory: ", trajectory.shape)

    smoothed_trajectory = smooth(trajectory)
    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory
    # Calculate newer transformation array
    transforms_smooth = transforms + difference

    # Write n_frames-1 transformed frames
    for j in range(1, n_frames - 2):
        frame = frames_list[j]

        # Extract transformations from the new transformation array
        dx = transforms_smooth[j, 0]
        dy = transforms_smooth[j, 1]
        da = transforms_smooth[j, 2]

        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (w, h))

        # Fix border artifacts
        frame_stabilized = fix_border(frame_stabilized)
        # Write the frame to the file
        # frame_out = cv2.hconcat([frame, frame_stabilized])

        # If the image is too big, resize it.
        # if frame_out.shape[1] >= 1920:
        #     frame_out = cv2.resize(frame_out, (int(frame_out.shape[1] / 2), int(frame_out.shape[0] / 2)))

        # plt.imshow(frame_out)
        # plt.show()
        # cv2.imwrite('./stabilized_frames/frame_out_radius{}_{}.jpg'.format(smoothing_radius, j), frame_out)

    return frame_stabilized


if __name__ == "__main__":
    frames_list = read_frames(source_path)
    # print(frames_list)
    n_frames = len(frames_list)
    h, w = frames_list[0].shape[0], frames_list[0].shape[1]
    prev = frames_list[0]
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    # Get stabilized images
    get_stabilized_frames()

