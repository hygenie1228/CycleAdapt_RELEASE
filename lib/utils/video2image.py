import os
import os.path as osp
import cv2
import subprocess
import argparse

def video_to_images(vid_file, img_folder=None, return_info=False):
    if img_folder is None:
        img_folder = osp.join('/tmp', osp.basename(vid_file).replace('.', '_'))
    
    os.makedirs(img_folder, exist_ok=True)

    command = ['ffmpeg',
               '-i', vid_file,
               '-f', 'image2',
               '-v', 'error',
               f'{img_folder}/%06d.png']
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

    print(f'Images saved to \"{img_folder}\"')

    return img_folder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, help='video path', required=True)
    args = parser.parse_args()
    
    img_folder = args.video.replace(args.video.split('/')[-1], '')
    img_folder = osp.join(img_folder, 'images')    

    video_to_images(args.video, img_folder)