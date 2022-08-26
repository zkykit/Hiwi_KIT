import argparse
from cProfile import label
import fnmatch
import glob
import json
import os
import shutil
import subprocess
import uuid
import sys
from joblib import delayed
from joblib import Parallel
import pandas as pd
###set up label list, named label_store
f = open('test_json.json') 
data = json.load(f)
label_list = []
i = 0
for i in range(0,len(data['tricks'])):
    label_name = data['tricks'][i]['labels']
    label_list.append(label_name)
    i=i+1
print(label_list)
def create_video_folders(data, output_dir, tmp_dir):#creat folders for save at some next steps
    """Creates a directory for each label name in the dataset."""#dataset是什么，自己的json吗
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)#this is path name
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)#this is also path name
   

    label_to_dir = {}
    for label_name_info in label_list:# dataset['label-name]应该是带有label name的一个list
        this_dir = os.path.join(output_dir, label_name_info)#this_dir是path+labelname
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)
        label_to_dir[label_name_info] = this_dir#this_dir赋值给label_to_dir
    return label_to_dir


def construct_video_filename(row, label_to_dir, trim_format='%06d'):#整数宽度=6，若不足6，左边补空格
    """Given a dataset row, this function constructs the 
       output filename for a given video.
    """
    #我要对basename进行更改
    #%的使用： 后者替代前者%s
    basename = '%s_%s_%s.mp4' % (row['video-id'],#尝试用test.py来修改
                                 trim_format % row['start-time'],
                                 trim_format % row['end-time'])
    output_filename = os.path.join(label_to_dir[row['label-name']], basename)#路径拼接
    return output_filename

sys.exit()
def download_clip(output_filename,#video saved in output_filename
                  start_time, end_time, 
                  tmp_dir='/tmp/kinetics',#这个路径应该是绝对路径
                  num_attempts=5,
                  url_base='https://www.youtube.com/watch?v='):
    """Download a video from youtube if exists and is not blocked.
    
    arguments:
    ---------
    video_identifier: str
        Unique YouTube video identifier (11 characters)
    output_filename: str
        File path where the video will be stored.
    start_time: float
        Indicates the begining time in seconds from where the video 
        will be trimmed.
    end_time: float
        Indicates the ending time in seconds of the trimmed video.
    """
    # Defensive argument checking.
    assert isinstance(output_filename, str), 'output_filename must be string'
    status = False

    # Construct command line for getting the direct video link.
    command = ['youtube-dl',
               '--quiet', '--no-warnings',
               '-f', 'mp4',
               '--get-url',
               '"%s"' % (url_base )]
    command = ' '.join(command)
    attempts = 0
    while True:
         try:
            direct_download_url = subprocess.check_output(command,
                                                          shell=True,
                                                          stderr=subprocess.STDOUT)
            direct_download_url = direct_download_url.strip().decode('utf-8')
         except subprocess.CalledProcessError as err:
            attempts += 1
            if attempts == num_attempts:
                return status, err.output
         else:
            break

    # Construct command to trim the videos (ffmpeg required) 构造修剪视频命令
    
    f = open('test_json.json') 
    data = json.load(f)
    print(type(data['tricks']))
    print(len(data['tricks']))
    direct_download_url = "https://cvhci.anthropomatik.kit.edu/~dschneider/jubot_demo_large/videos/2022_03_10_S01_SC00_T01.mp4"

    i=0
    for i in range(0,len(data['tricks'])):

        start_time = data['tricks'][i]['start']
        end_time = data['tricks'][i]['end']
        print(start_time, end_time)
        i=i+1
    #path = /video/test[i]
        command = ['ffmpeg',
                        '-ss', str(start_time),
                        '-t', str(end_time-start_time),
                        '-i', "'%s'" % direct_download_url,
                        '-c:v', 'copy', '-c:a', 'copy',
                        '-threads', '1',
                        '-loglevel', 'panic',
                        '"%s"' % output_filename]
        command = ' '.join(str(n) for n in command)#表示command之间有一个空格
        try:
            output = subprocess.check_output(command, shell=True,
                                                    stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            return status, err.output

    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    return status, 'Downloaded'


def download_clip_wrapper(row, label_to_dir, trim_format, tmp_dir):#下载剪辑包装器
    """Wrapper for parallel processing purposes."""
    output_filename = construct_video_filename(row, label_to_dir,
                                               trim_format)
    clip_id = os.path.basename(output_filename).split('.mp4')[0]#.mp4为分隔符,后的第一个字符
    if os.path.exists(output_filename):
        status = tuple([clip_id, True, 'Exists'])
        return status

    downloaded, log = download_clip(row['video-id'], output_filename,
                                    row['start-time'], row['end-time'],
                                    tmp_dir=tmp_dir)
    status = tuple([clip_id, downloaded, log])
    return status

'''
def parse_kinetics_annotations(input_csv):#read csv,我需要load json
    """Returns a parsed DataFrame.
    
    arguments:
    ---------
    input_csv: str
        Path to CSV file containing the following columns:
          'YouTube Identifier,Start time,End time,Class label'
    returns:
    -------
    dataset: DataFrame
        Pandas with the following columns:
            'video-id', 'start-time', 'end-time', 'label-name'
    """
    df = pd.read_csv(input_csv)
    df.rename(columns={'youtube_id': 'video-id',
                       'time_start': 'start-time',
                       'time_end': 'end-time',
                       'label': 'label-name',
                       'is_cc': 'is-cc'}, inplace=True)
    return df
'''

def main(output_dir,
         trim_format='%06d', num_jobs=24, tmp_dir='/tmp/kinetics'):

    # Reading and parsing Kinetics.
    #dataset = parse_kinetics_annotations(input_csv)
 
    # Creates folders where videos will be saved later.
    label_to_dir = create_video_folders(output_dir, tmp_dir)

    # Download all clips.
    if num_jobs==1:
        status_lst = []
        for i, row in dataset.iterrows():
            status_lst.append(download_clip_wrapper(row, label_to_dir, 
                                                    trim_format, tmp_dir))
    else:
        status_lst = Parallel(n_jobs=num_jobs)(delayed(download_clip_wrapper)(
            row, label_to_dir,
            trim_format, tmp_dir) for i, row in dataset.iterrows())

    # Clean tmp dir.
    shutil.rmtree(tmp_dir)

    # Save download report.
    with open('download_report.json', 'w') as fobj:
        fobj.write(json.dumps(status_lst))


if __name__ == '__main__':
    description = 'Helper script for downloading and trimming kinetics videos.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('input_csv', type=str,
                   help=('CSV file containing the following format: '
                         'YouTube Identifier,Start time,End time,Class label'))
    p.add_argument('output_dir', type=str,
                   help='Output directory where videos will be saved.')
    p.add_argument('-f', '--trim-format', type=str, default='%06d',
                   help=('This will be the format for the '
                         'filename of trimmed videos: '
                         'videoid_%0xd(start_time)_%0xd(end_time).mp4'))
    p.add_argument('-n', '--num-jobs', type=int, default=24)
    p.add_argument('-t', '--tmp-dir', type=str, default='/tmp/kinetics')
    main(**vars(p.parse_args()))