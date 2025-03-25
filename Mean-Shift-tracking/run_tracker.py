import time

import cv2

from sequence_utils import VOTSequence
#from ncc_tracker_example import NCCTracker, NCCParams
from ms_tracker import MeanShiftTracker, MSParams


# set the path to directory where you have the sequences
dataset_path = 'C:\\Users\\marko\\Desktop\\Computer_vision\\Advanced-methods-in-computer-vision-assignments\\Mean-Shift-tracking\\vot2014' # TODO: set to the dataet path on your disk
sequence ='ball'  # choose the sequence you want to test

# visualization and setup parameters
win_name = 'Tracking window'
reinitialize = True
show_gt = False
video_delay = 15
font = cv2.FONT_HERSHEY_PLAIN

# create sequence object
sequence = VOTSequence(dataset_path, sequence)
init_frame = 0

n_failures = 0
# create parameters and tracker objects
# parameters = NCCParams()
# tracker = NCCTracker(parameters)
parameters = MSParams()
tracker = MeanShiftTracker(parameters)

time_all = 0

# initialize visualization window
sequence.initialize_window(win_name)
# tracking loop - goes over all frames in the video sequence
frame_idx = 0
while frame_idx < sequence.length():
    img = cv2.imread(sequence.frame(frame_idx))
    # initialize or track
    if frame_idx == init_frame:
        # initialize tracker (at the beginning of the sequence or after tracking failure)
        #print(sequence.get_annotation(frame_idx, type='rectangle'))
        t_ = time.time()
        tracker.initialize(img, sequence.get_annotation(frame_idx, type='rectangle'))
        time_all += time.time() - t_
        predicted_bbox = sequence.get_annotation(frame_idx, type='rectangle')
    else:
        # track on current frame - predict bounding box
        t_ = time.time()
        predicted_bbox = tracker.track(img)
        time_all += time.time() - t_

    # calculate overlap (needed to determine failure of a tracker)
    gt_bb = sequence.get_annotation(frame_idx, type='rectangle')
    o = sequence.overlap(predicted_bbox, gt_bb)

    # draw ground-truth and predicted bounding boxes, frame numbers and show image
    if show_gt:
        sequence.draw_region(img, gt_bb, (0, 255, 0), 1)
    sequence.draw_region(img, predicted_bbox, (0, 0, 255), 2)
    sequence.draw_text(img, '%d/%d' % (frame_idx + 1, sequence.length()), (25, 25))
    sequence.draw_text(img, 'Fails: %d' % n_failures, (25, 55))
    sequence.show_image(img, video_delay)

    if o > 0 or not reinitialize:
        # increase frame counter by 1
        frame_idx += 1
    else:
        # increase frame counter by 5 and set re-initialization to the next frame
        frame_idx += 5
        init_frame = frame_idx
        n_failures += 1

print('Tracking speed: %.1f FPS' % (sequence.length() / time_all))
print('Tracker failed %d times' % n_failures)
