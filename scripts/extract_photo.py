import os 
import cv2
import PIL 
import time 
import glob 
import json 
import pytz
import argparse 
import subprocess 

import numpy as np 
import pandas as pd 
import xml.etree.ElementTree as ET

from PIL import Image 
from datetime import datetime, timedelta, timezone

# Extract start time using Exiftool
def extract_start_time(video_file):
    """Extract the start time from a GoPro video file using ExifTool."""
    try:
        exiftool_path = './exiftool/exiftool-12.92_64/exiftool.exe'
        exiftool_command = [exiftool_path, "-CreateDate", "-j", video_file]
        result = subprocess.run(exiftool_command, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, check=True)
        
        metadata = json.loads(result.stdout)[0]
        create_date_str = metadata.get('CreateDate')
        
        if not create_date_str:
            raise ValueError("CreateDate not found in video metadata")
            
        return datetime.strptime(create_date_str, '%Y:%m:%d %H:%M:%S')
    
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error extracting start time: {e}")
        return None

def video_to_frames(input_loc, output_loc, combined_video):
    """Extracts one frame per second from a video and saves them with timestamps."""
    start_time = extract_start_time(input_loc)
    if not start_time:
        print("Aborting due to missing start time")
        return

    os.makedirs(output_loc, exist_ok=True)
    
    try:
        cap = cv2.VideoCapture(combined_video)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {combined_video}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(round(fps)) if fps > 0 else 1
        print(f"Processing video with {fps:.2f} FPS (saving every {frame_interval} frames)")

        saved_count = 0
        processing_start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame % frame_interval == 0:
                timestamp = start_time + timedelta(seconds=saved_count)
                filename = timestamp.strftime('frame_%Y-%m-%d_%H-%M-%S.jpg')
                cv2.imwrite(os.path.join(output_loc, filename), frame)
                saved_count += 1

        print(f"Successfully saved {saved_count} frames in {time.time()-processing_start:.1f} seconds")

    except Exception as e:
        print(f"Processing failed: {e}")
    finally:
        if 'cap' in locals():
            cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract timestamped frames from GoPro video',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s','--source_video', help='Original video for metadata extraction')
    parser.add_argument('-o','--output_dir', help='Directory to save timestamped frames')
    parser.add_argument('-p','--processed_video', help='Actual video to process (could be processed version)')
    
    args = parser.parse_args()
    
    video_to_frames(args.source_video, args.output_dir, args.processed_video)