"""
This script kicks off the video processing pipeline for 
phacotrainer which consists of several steps:

1. Collect newly uploaded (timestamp > last_run_timestamp) videos from firestore 
    - videoId
    - uid
    - filename
    - creationDate (should be > last_run_timestamp)
    - processed (should be False)

2. Download each video file from GCP blob storage
    - construct storage path using /phacotrainer-app-path/{uid}/{filename}

3. Convert each video to its corresponding .h5 file using preprocess_video.py

4. Produce model output for each .h5 file for the video using model.py

5. Write the model output for videoId to modelOutput table in firestore

6. Update videos table "processed" field for each successful video in firestore

7. Delete locally downloaded videos after processing all

@author: Simmi Sen
@date: 05/31/2022
"""

