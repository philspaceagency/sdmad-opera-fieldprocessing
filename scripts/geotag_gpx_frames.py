import os
import cv2
import time
import glob 
import pytz 
import subprocess

import numpy as np
import pandas as pd

import xml.etree.ElementTree as ET

import PIL
from PIL import Image
from PIL import PngImagePlugin
from datetime import datetime, timedelta, timezone

def convert_utc_to_pst(utc_time_str):
    utc_time = datetime.strptime(utc_time_str, '%Y-%m-%dT%H:%M:%SZ')
    utc_zone = pytz.timezone('UTC')
    pst_zone = pytz.timezone('Asia/Manila')
    utc_time = utc_zone.localize(utc_time)
    pst_time = utc_time.astimezone(pst_zone)
    return pst_time.strftime('%Y-%m-%dT%H:%M:%S%z')

def read_gpx(gpx_file):
    tree = ET.parse(gpx_file)
    root = tree.getroot()

    namespace = {
        'default': 'http://www.topografix.com/GPX/1/1',
        'gpxtpx': 'http://www.garmin.com/xmlschemas/TrackPointExtension/v1',
        'gpxx': 'http://www.garmin.com/xmlschemas/GpxExtensions/v3'
    }

    data = {'lat': [], 'lon': [], 'time': [], 'depth': []}

    for trkpt in root.findall('.//default:trkpt', namespace):
        lat = trkpt.get('lat')
        lon = trkpt.get('lon')
        time_utc = trkpt.find('default:time', namespace).text if trkpt.find('default:time', namespace) is not None else None
        time_pst = convert_utc_to_pst(time_utc) if time_utc else None

        depth = None
        ext = trkpt.find('default:extensions', namespace)
        if ext is not None:
            gpxtpx = ext.find('gpxtpx:TrackPointExtension/gpxtpx:depth', namespace)
            gpxx = ext.find('gpxx:TrackPointExtension/gpxx:Depth', namespace)
            if gpxtpx is not None:
                depth = float(gpxtpx.text)
            elif gpxx is not None:
                depth = float(gpxx.text)

        data['lat'].append(lat)
        data['lon'].append(lon)
        data['time'].append(time_pst)
        data['depth'].append(depth)

    return pd.DataFrame(data)

def convert_to_iso_format(original_timestamp):
    dt = datetime.strptime(original_timestamp, '%Y-%m-%d %H-%M-%S')
    return dt.strftime('%Y-%m-%dT%H:%M:%S') + '+0800'

def interpolate_coordinates(frame_time, df_gpx):
    before = df_gpx[df_gpx['time'] <= frame_time]
    after = df_gpx[df_gpx['time'] > frame_time]

    if before.empty or after.empty:
        return np.nan, np.nan

    before = before.iloc[-1]
    after = after.iloc[0]

    time_diff = (after['time'] - before['time']).total_seconds()
    time_ratio = (frame_time - before['time']).total_seconds() / time_diff

    lat = before['lat'] + time_ratio * (after['lat'] - before['lat'])
    lon = before['lon'] + time_ratio * (after['lon'] - before['lon'])

    return lat, lon

def geotag_images(csv_file, images_dir):
    with open(csv_file, 'r') as file:
        lines = file.readlines()[1:]

    for line in lines:
        frame_datetime, frame_filename, latitude, longitude, frame_dir = line.strip().split(',')

        if frame_filename and latitude and longitude:
            image_path = os.path.join(images_dir, frame_filename)
            if os.path.exists(image_path):
                command = [
                    './exiftool/exiftool-12.92_64/exiftool.exe',
                    f'-GPSLatitude={latitude}',
                    f'-GPSLongitude={longitude}',
                    '-GPSLatitudeRef=N',
                    '-GPSLongitudeRef=E',
                    image_path
                ]
                try:
                    subprocess.run(command, check=True)
                    print(f"Geotagged {frame_filename} with coordinates ({latitude}, {longitude})")
                except subprocess.CalledProcessError as e:
                    print(f"Error geotagging {frame_filename}: {e}")

if __name__ == "__main__":
    gpx_file_path = './GPX/13MAY2025.GPX'
    frame_dir = './Output'
    out_dir = './Test'
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])

    gpx_df = read_gpx(gpx_file_path)
    df_ = gpx_df[gpx_df['depth'].notnull()]
    df_gpx = df_[df_['time'].notnull()]
    df_gpx.loc[:, 'lat'] = pd.to_numeric(df_gpx['lat'], errors='coerce')
    df_gpx.loc[:, 'lon'] = pd.to_numeric(df_gpx['lon'], errors='coerce')
    df_gpx.loc[:, 'time'] = pd.to_datetime(df_gpx['time'].str.replace('Z', '+00:00'))
    df_gpx = df_gpx.sort_values(by='time')

    stamp = []
    for frames in frame_files:
        date = frames.split("_")[1]
        time = frames.split("_")[2].split(".")[0]
        timestamp = date + " " + time
        new_timestamp = convert_to_iso_format(timestamp)
        stamp.append(new_timestamp)

    df_frames = pd.DataFrame({
        'frame_datetime': stamp,
        'frame_filename': frame_files
    })
    df_frames['frame_datetime'] = pd.to_datetime(df_frames['frame_datetime'])
    df_frames = df_frames.sort_values(by='frame_datetime')
    df_frames['latitude'] = np.nan
    df_frames['longitude'] = np.nan

    for idx, row in df_frames.iterrows():
        lat, lon = interpolate_coordinates(row['frame_datetime'], df_gpx)
        df_frames.at[idx, 'latitude'] = lat
        df_frames.at[idx, 'longitude'] = lon

    df_frames['frame_dir'] = frame_dir + '/' + df_frames['frame_filename']
    output_csv = 'D:/PhilSA/OpERA/_FieldExtraction/frame_data.csv'
    df_frames.to_csv(output_csv, index=False)

    geotag_images(output_csv, frame_dir)