import os
import json
import subprocess
import numpy as np

from typing import List, Dict, Optional
from gcp_client import GCPClient
from preprocess import H5Converter
from model import PhacotrainerModel


VIDEO_DIR = './videos'
PREPROCESS_DIR = './h5_files'
MODEL_OUTPUT_DIR = './model_outputs'
THUMBNAIL_PATH = './thumbnails'


def remove_files(cleanup_dirs: Optional[List[str]]) -> None:
    """
    Handles removing files in the temporary directories once processing complete.

    Args:
        cleanup_dirs: Directory paths to remove all files from once script complete.
        type: List[str]
    """
    if not cleanup_dirs:
        cleanup_dirs = [
            VIDEO_DIR,
            PREPROCESS_DIR,
            MODEL_OUTPUT_DIR,
        ]

    for dir in cleanup_dirs:
        for filename in os.listdir(dir):
            os.remove(os.path.join(dir, filename))


def get_raw_outputs(
        preds: List[int], timestamps: np.ndarray, label_names: List[str]
    ) -> Dict:
    """
    Parses the predictions and timestamps from the model into 
    a dictionary.

    Args:
        preds: Model predictions for each timestamp as a list of int
        type: List[int]
        
        timestamps: Start and end times over which a prediction is made.
        type: np.ndarray
    Returns:
        raw_outputs: Unified dictionary containing each entry with start/end time and prediction
        type: Dict
    """

    raw_outputs = {"results": []}

    for i, val in enumerate(preds):
        label = label_names[int(val)]

        start_idx = i
        end_idx = i + 1 if (i + 1) < len(timestamps) else i

        raw_outputs["results"].append({
            "start_time": int(timestamps[start_idx]),
            "end_time": int(timestamps[end_idx]),
            "label": label
        })
    
    return raw_outputs


def process_raw_outputs(raw_outputs: Dict, no_label: str) -> Dict:
    """
    Filters raw_outputs to not contain 'No Label' predictions.

    Args:
        raw_outputs: Raw predictions from the ML model
        type: Dict
        no_label: The label name assigned when no label is able to be predicted
    Returns:
        processed_outputs: Unified dictionary without 'no label' predictions
        type: Dict
    """

    processed_outputs = {"results": []}

    for output in raw_outputs["results"]:
        if output["label"] != no_label:
            processed_outputs["results"].append(output)
    
    return processed_outputs


def compute_insights(raw_outputs: Dict, label_names: List[str]) -> Dict:
    """
    Gets the insights on time taken per step from the raw_outputs 
    and stores this information as a dictionary.
    """

    insights = {
        label: {"count": 0, "timeTaken": 0} 
        for label in label_names
    }

    for step in raw_outputs["results"]:
        label = step["label"]
        time_taken = step["end_time"] - step["start_time"]

        insights[label]["count"] += 1
        insights[label]["timeTaken"] += time_taken
    
    return insights


def preprocess_videos(downloaded_videos: List[str]) -> List[str]:
    """
    Preprocesses videos using cv2 over downloaded_videos.

    Args:
        downloaded_videos: Filepaths to all downloaded unprocessed videos
        type: List[str]
    Returns:
        preprocessed_files: Filepaths to the successfully preprocessed files
        type: List[str]
    """

    converter = H5Converter()

    print(downloaded_videos)

    preprocessed_files = []
    for video_file in downloaded_videos:
        print("Preprocessing video at {}".format(video_file))

        preprocessed_file, _, _ = converter.convert(video_file)

        preprocessed_files.append(preprocessed_file)

    return preprocessed_files


def run_model(preprocessed_videos: List[str]) -> List[str]:
    """
    Runs the model over the preprocessed video filepaths.
    Writes model outputs to MODEL_OUTPUT_DIR.

    Args:
        preprocessed_videos: List of paths to preprocessed videos in h5_files folder.
        type: List[str]

        client: GCP client with methods for reading / writing to firestore tables.
        type: GCPClient
    Returns:
        predicted_videos: List of video_ids for videos that model predictions were made.
        type: List[str]
    """

    model = PhacotrainerModel()

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    predicted_videos = []
    print("Running model on " + str(len(preprocessed_videos)) + " preprocessed videos")
    for preprocessed_file in preprocessed_videos:
        filename = preprocessed_file.split('/')[-1]
        video_id = filename.split('.')[0]

        if os.path.exists(f'{MODEL_OUTPUT_DIR}/{video_id}_raw.json'):
            print("Predictions already exist for {}".format(video_id))
            continue
            
        print("Making predictions for: {} ...".format(video_id))
        preds, timestamps = model.predict(filename)

        raw_outputs = get_raw_outputs(preds, timestamps, model.label_names)
        processed_outputs = process_raw_outputs(raw_outputs, model.label_names[-1])
        insights = compute_insights(raw_outputs, model.label_names)

        with open(f'{MODEL_OUTPUT_DIR}/{video_id}_raw.json', 'w') as raw_output_file:
            raw_output_file.write(json.dumps(raw_outputs))
        
        with open(f'{MODEL_OUTPUT_DIR}/{video_id}_processed.json', 'w') as processed_output_file:
            processed_output_file.write(json.dumps(processed_outputs))

        with open(f'{MODEL_OUTPUT_DIR}/{video_id}_insights.json', 'w') as insights_file:
            insights_file.write(json.dumps(insights))

        predicted_videos.append(video_id)

    return predicted_videos


def write_model_outputs(video_ids: List[str], client: GCPClient):
    """
    Save relevant model outputs and insights to firestore tables.

    Args:
        video_ids: List of video ids to write outputs to firestore
        type: List[str]

        client: GCP client used to read/write to firestore
        type: GCPClient
    Returns:
        written_outputs: List of video_ids that were successfully written to firestore
        type: List[str]
    """
    
    written_outputs = []

    for video_id in video_ids:
        
        raw_output_file = open(f'{MODEL_OUTPUT_DIR}/{video_id}_raw.json', 'r')
        processed_output_file = open(f'{MODEL_OUTPUT_DIR}/{video_id}_processed.json', 'r')
        insights_file = open(f'{MODEL_OUTPUT_DIR}/{video_id}_insights.json', 'r')

        client.write_model_outputs(
            video_id=video_id, 
            raw_output=raw_output_file.read(), 
            processed_output=processed_output_file.read(), 
            insights=insights_file.read(),
            verbose=True   
        )

        written_outputs.append(video_id)

        raw_output_file.close()
        processed_output_file.close()
        insights_file.close()
    
    return written_outputs


def publish_insights(video_ids: List[str], client: GCPClient):
    """
    Handles publishing the insights over the video_ids to the insights table.

    Args:
        video_ids: List of video ids to write outputs to firestore
        type: List[str]

        client: GCP client used to read/write to firestore
        type: GCPClient
    """
    
    insights_dict = client.get_video_insights(video_ids)
    client.write_insights(insights_dict)

    return video_ids


def update_video_metadata(video_ids: List[str], client: GCPClient):
    """
    Handles publishing the insights over the video_ids to the insights table.

    Args:
        video_ids: List of video ids to write outputs to firestore
        type: List[str]

        client: GCP client used to read/write to firestore
        type: GCPClient
    """

    insights_dict = client.get_video_insights(video_ids)

    video_db = client.videos_table
    for video_id in insights_dict:
        print("Updating metadata for {}...".format(video_id))

        surgery_steps = []
        for step, metadata in insights_dict[video_id].items():
            if metadata['timeTaken'] > 100:
                surgery_steps.append(step)
        
        # docs = video_db.where(u'videoId', u'==', int(video_id)).stream()

        video = client._get_video(video_id)
        curr_doc = video_db.document(video.doc_id).get()
        metadata = curr_doc.to_dict().get("metadata", {})
        metadata["surgery_steps"] = surgery_steps
        video_db.document(video.doc_id).update({u"metadata": metadata, u"processed": True})
    
    return video_ids


def create_thumbnails(video_path: str, thumbnail_path: str, client: GCPClient):
    """
    Creates and saves thumbnails for all videos in `video_path`.

    Args:
        video_path: path to phacotrainer videos being processed
        type: str

        thumbnail_path: path to save created thumbnails
        type: str
    """

    if not os.path.exists(video_path):
        print('Invalid path to videos {}'.format(video_path))
    
    if not os.path.exists(thumbnail_path):
        os.makedirs(thumbnail_path, exist_ok=True)
    
    for uid in os.listdir(video_path):
        for filename in os.listdir(f'{video_path}/{uid}'):
            if filename.startswith('.'):
                continue

            os.makedirs(f'{thumbnail_path}/{uid}', exist_ok=True)

            video_id = filename.split('.')[0]

            video_input_path = os.path.join(f'{video_path}/{uid}', filename)
            thumb_output_path = os.path.join(f'{thumbnail_path}/{uid}', video_id + '.jpg')

            subprocess.call(['ffmpeg', '-i', video_input_path, '-ss',
                            '00:00:00.000', '-vframes', '1', thumb_output_path])

    for uid in os.listdir(thumbnail_path):
        for filename in os.listdir(f'{thumbnail_path}/{uid}'):
            client.publish_thumbnail(f'{thumbnail_path}/{uid}/{filename}')