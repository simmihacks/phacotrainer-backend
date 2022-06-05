"""
This script takes a single video and uses the CV2 library to convert it 
to the expected model input, which is a .h5 file with dimensions 
[num_frames, 256, 456, 3].

@author: Simmi Sen
@date: 05/31/2022
"""

import os
import cv2
import h5py  
import numpy as np


class H5Converter(object):

    def __init__(self):
        self.videos_path = "data"
        self.output_path = "h5_files"
        self.make_png = False
        self.make_h5 = True
        self.offset = 0

    def _load(self, filename):
        """
        Handles loading the video file using cv2 library
        from the provided filepath.
        """
        filepath = os.path.join(self.videos_path, filename)

        try:
            cap = cv2.VideoCapture(filepath)
            fps = cap.get(cv2.CAP_PROP_FPS) # should be 90
        except:
            print("Unable to load video {} with cv2".format(filepath))
            return None
        
        return cap, fps

    def convert(self, filename):
        """
        Handles converting the file at `filepath` to a numpy
        array of frames sampled at the computed fps and saved 
        as a .h5 file.
        """

        frames = []
        timestamps = []
        cap, fps = self._load(filename)

        msframenum = 0

        while cap.isOpened():
            frame_exists, curr_frame = cap.read()

            if frame_exists:
                msframenum += 1 
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

                # sample video every second. With offset we can choose to sample 
                # at frame 25/50/75, vs 35/60/85 etc
                if msframenum + self.offset % round(fps) == 1:   
                    timestamps.append(timestamp) 
                    curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
                    frames.append(curr_frame)
        
        video_frames = np.stack(frames)

        output_filepath = os.join(self.output_path, filename.split('.')[0] + '.h5')
        with h5py.File(output_filepath, 'w') as hf:
            hf.create_dataset('frames', data=video_frames, compression='gzip')
            hf.create_dataset('timestamps', data=timestamps)
        
        cap.release()

        return video_frames, np.array(timestamps)
