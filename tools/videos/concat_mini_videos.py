import argparse

from avcv.all import *

parser = argparse.ArgumentParser()
parser.add_argument("path_to_folder")
parser.add_argument("output_path")
args = parser.parse_args()


def format_second_to_hhmmss(second):
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


# Use ffmpeg concat a list of video
videos = glob(args.path_to_folder + "/*.mp4")
videos = list(sorted(videos))


# Use ffmpeg concat a list of video
def concat_two_video(video_1, video_2, out):
    mmcv.mkdir_or_exist(os.path.dirname(out))
    if video_1 == out:
        os.system("cp {} {}".format(video_1, video_1 + ".tmp"))
        video_1 = video_1 + ".tmp"
    cmd = f"/usr/bin/ffmpeg -y -i {video_1} -i {video_2} -filter_complex '[0:v:0][1:v:0]concat=n=2:v=1:a=0' -c:v libx264 {out}"
    print(cmd)
    os.system(cmd)
    # os.remove(video_1)


def concat_n_video(video_list, out):
    mmcv.mkdir_or_exist(os.path.dirname(out))
    # ffmpeg concat a list of video and keep the audio
    cmd = f"/usr/bin/ffmpeg -y -i {' -i '.join(video_list)} -filter_complex '[0:v:0]concat=n={len(video_list)}:v=1:a=0' -c:v libx264 -c:a copy {out}"
    print(cmd)
    os.system(cmd)


def concat_n_audio(audio_list, out):
    mmcv.mkdir_or_exist(os.path.dirname(out))
    # ffmpeg concat a list of audio to out
    cmd = f"/usr/bin/ffmpeg -y -i {' -i '.join(audio_list)} -filter_complex '[0:a:0]concat=n={len(audio_list)}:v=0:a=1' -c:v libx264 -c:a aac {out}"
    print(cmd)
    os.system(cmd)


def get_audio_path(video_path, audio_dir):
    return os.path.join(audio_dir, os.path.basename(video_path).replace(".mp4", ".wav"))


for i in range(0, len(videos), 10):
    _videos = videos[i : i + 10]
    _audios = [get_audio_path(video, "data/private_test/songs") for video in _videos]
    concat_n_audio(_audios, "tmp.wav")
    out_path = args.output_path.replace(".mp4", f"_{i:03d}_{i+10:03d}.mp4")
    with open(out_path.replace(".mp4", ".txt"), "w") as f:
        current_second = 0
        for i, video in enumerate(_videos):
            file_name = video.split("/")[-1].split(".")[0]
            video = mmcv.VideoReader(video)
            video_len_in_second = len(video) / video.fps
            current_str = format_second_to_hhmmss(current_second)
            f.write(f"|{i:03d}-{current_str}")
            if i % 5 == 0 and i > 0:
                f.write("\n")
            current_second += video_len_in_second
    concat_n_video(_videos, out_path)
    os.rename(out_path, out_path.replace(".mp4", "_tmp.mp4"))
    # Concat video/audio
    cmd = f"/usr/bin/ffmpeg -y -i {out_path.replace('.mp4', '_tmp.mp4')} -i tmp.wav -c:v copy -c:a copy {out_path}"
    print(cmd)
    os.system(cmd)
    os.remove(out_path.replace(".mp4", "_tmp.mp4"))
    # os.remove('tmp.wav')
    break
