from avcv.all import *
def concat_video(video1, video2, audio, output):
    """Concatenate two videos"""
    # Load video
    video1 = mmcv.VideoReader(video1)
    video2 = mmcv.VideoReader(video2)
    # Concatenate
    # video = np.concatenate((video1, video2), axis=0)
    concat_frames = []
    for frame1, frame2 in zip(video1, video2):
        concat_frames.append(np.concatenate((frame1, frame2), axis=0))
    print('->output', output)
    mmcv.mkdir_or_exist(osp.dirname(output))
    tmp_output = output.replace('.mp4', '_tmp.mp4')
    images_to_video(concat_frames, tmp_output, fps=video1.fps, output_size=(video1.width, video1.height*2))
    # Using ffmpeg concat audio and output video together
    # ffmpeg -i video.mp4 -i audio.mp3 -c:v copy -c:a aac -strict experimental output.mp4
    os.system('ffmpeg -y -i {} -i {} -c:v copy -c:a aac -strict experimental {}'.format(tmp_output, audio, output))
    os.remove(tmp_output)
    # # Write video
    # video = mmcv.VideoWriter(output, fps=video1.fps)
    # for frame in video:
    #     video.write(frame)
    # video.release()
    
    




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('video_folder_1')
    parser.add_argument('video_folder_2')
    parser.add_argument('audio_folder')
    parser.add_argument('fuse_folder')
    args = parser.parse_args()
    
    video_paths_1 = glob(osp.join(args.video_folder_1, '*.mp4'))
    # video_paths_2 = glob(osp.join(args.video_folder_2, '*.mp4'))
    
    
    for video_path_1 in video_paths_1:
        audio_path = osp.join(args.audio_folder, osp.basename(video_path_1).replace('.mp4', '.wav'))
        video_paths_2 = glob(osp.join(args.video_folder_2, osp.basename(video_path_1)))
        out_path = osp.join(args.fuse_folder, osp.basename(video_path_1))
        concat_video(video_path_1, video_paths_2[0], audio_path, out_path)