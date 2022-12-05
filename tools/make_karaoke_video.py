import os
import os.path as osp
from glob import glob
from typing import Tuple

import cv2
import mmcv
import numpy as np
import torch
import torchaudio
from moviepy.editor import AudioFileClip, VideoFileClip
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def convert_vietnamese_text_to_english_text(text: str):

    """
    Convert Vietnamese text to English text
    """
    text = text.lower()
    text = text.replace("á", "a")
    text = text.replace("à", "a")
    text = text.replace("ả", "a")
    text = text.replace("ã", "a")
    text = text.replace("ạ", "a")
    text = text.replace("â", "a")
    text = text.replace("ấ", "a")
    text = text.replace("ầ", "a")
    text = text.replace("ẩ", "a")
    text = text.replace("ẫ", "a")
    text = text.replace("ậ", "a")
    text = text.replace("ă", "a")
    text = text.replace("ắ", "a")
    text = text.replace("ằ", "a")
    text = text.replace("ẳ", "a")
    text = text.replace("ẵ", "a")
    text = text.replace("ặ", "a")
    text = text.replace("đ", "d")
    text = text.replace("é", "e")
    text = text.replace("è", "e")
    text = text.replace("ẻ", "e")
    text = text.replace("ẽ", "e")
    text = text.replace("ẹ", "e")
    text = text.replace("ê", "e")
    text = text.replace("ế", "e")
    text = text.replace("ề", "e")
    text = text.replace("ể", "e")
    text = text.replace("ễ", "e")
    text = text.replace("ệ", "e")
    text = text.replace("í", "i")
    text = text.replace("ì", "i")
    text = text.replace("ỉ", "i")
    text = text.replace("ĩ", "i")
    text = text.replace("ị", "i")
    text = text.replace("ó", "o")
    text = text.replace("ò", "o")
    text = text.replace("ỏ", "o")
    text = text.replace("õ", "o")
    text = text.replace("ọ", "o")
    text = text.replace("ô", "o")
    text = text.replace("ố", "o")
    text = text.replace("ồ", "o")
    text = text.replace("ổ", "o")
    text = text.replace("ỗ", "o")
    text = text.replace("ộ", "o")
    text = text.replace("ơ", "o")
    text = text.replace("ớ", "o")
    text = text.replace("ờ", "o")
    text = text.replace("ở", "o")
    text = text.replace("ỡ", "o")
    text = text.replace("ợ", "o")
    text = text.replace("ú", "u")
    text = text.replace("ù", "u")
    text = text.replace("ủ", "u")
    text = text.replace("ũ", "u")
    text = text.replace("ụ", "u")
    text = text.replace("ư", "u")
    text = text.replace("ứ", "u")
    text = text.replace("ừ", "u")
    text = text.replace("ử", "u")
    text = text.replace("ữ", "u")
    text = text.replace("ự", "u")
    text = text.replace("ý", "y")
    text = text.replace("ỳ", "y")
    text = text.replace("ỷ", "y")
    text = text.replace("ỹ", "y")
    text = text.replace("ỵ", "y")
    text = text.replace("Á", "A")
    text = text.replace("À", "A")
    text = text.replace("Ả", "A")
    text = text.replace("Ã", "A")
    text = text.replace("Ạ", "A")
    text = text.replace("Â", "A")
    text = text.replace("Ấ", "A")
    text = text.replace("Ầ", "A")
    text = text.replace("Ẩ", "A")
    text = text.replace("Ẫ", "A")
    text = text.replace("Ậ", "A")
    text = text.replace("Ă", "A")
    text = text.replace("Ắ", "A")
    text = text.replace("Ằ", "A")
    text = text.replace("Ẳ", "A")
    text = text.replace("Ẵ", "A")
    text = text.replace("Ặ", "A")
    text = text.replace("Đ", "D")
    text = text.replace("É", "E")
    text = text.replace("È", "E")
    text = text.replace("Ẻ", "E")
    text = text.replace("Ẽ", "E")
    text = text.replace("Ẹ", "E")
    text = text.replace("Ê", "E")
    text = text.replace("Ế", "E")
    text = text.replace("Ề", "E")
    text = text.replace("Ể", "E")
    text = text.replace("Ễ", "E")
    text = text.replace("Ệ", "E")
    text = text.replace("Í", "I")
    text = text.replace("Ì", "I")
    text = text.replace("Ỉ", "I")
    text = text.replace("Ĩ", "I")
    text = text.replace("Ị", "I")
    text = text.replace("Ó", "O")
    text = text.replace("Ò", "O")
    text = text.replace("Ỏ", "O")
    text = text.replace("Õ", "O")

    return text


def torch_load_audio(audio_path, sr=16000):
    """
    Load audio using torchaudio
    """
    audio = torchaudio.load(audio_path)[0]
    return audio.numpy()


def generate_karaoke_video(audio_path, lyrics_path, output_video_path, fps=120):
    """
    Generate karaoke video from audio and lyrics
    
    """
    # Load audio and lyrics
    audio = torch_load_audio(audio_path, sr=16000)

    lyrics = mmcv.load(lyrics_path)

    for line in lyrics:
        line["s"] = line["s"] / 1000
        line["e"] = line["e"] / 1000
        for word in line["l"]:
            word["s"] = word["s"] / 1000
            word["e"] = word["e"] / 1000

    start_time = 0  # lyrics.pop(0)['s']
    end_time = lyrics.pop(-1)["e"]
    num_frames = int((end_time - start_time) * fps)

    frames = np.zeros((num_frames, 100, 1500, 3), dtype=np.uint8)

    for i, line in enumerate(lyrics):
        start_line_frame = int((line["s"] - start_time) * fps)
        end_line_frame = int((line["e"] - start_time) * fps)

        text_line = " ".join([word["d"] for word in line["l"]])
        first_frame_line = frames[start_line_frame].copy() * 0
        # Puttext to first frame using opencv
        text_line = convert_vietnamese_text_to_english_text(text_line)
        cv2.putText(
            first_frame_line,
            text_line,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Print text and time
        print(
            text_line, line["s"], line["e"], start_line_frame, end_line_frame
        )
        frames[start_line_frame:end_line_frame] = first_frame_line
        last_modified_frame = start_line_frame
        for j, word in enumerate(line["l"]):
            # Get start and end frame of word
            start_word_frame = int((word["s"] - start_time) * fps)
            end_word_frame = int((word["e"] - start_time) * fps)

            current_text_line = " ".join(
                [word["d"] for word in line["l"][: j + 1]]
            )
            current_text_line = convert_vietnamese_text_to_english_text(
                current_text_line
            )
            # Puttext to first frame of this word  using opencv
            # first_word_frame = frames[start_word_frame]
            middle_word_frame = frames[
                start_word_frame + (end_word_frame - start_word_frame)// 2
            ].copy()
            cv2.putText(
                middle_word_frame,
                current_text_line,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3,
            )
            # start_word_frame = max(last_modified_frame, start_word_frame)
            frames[start_word_frame:end_word_frame] = middle_word_frame
            # last_modified_frame = end_word_frame


    # Generate video from frames using opencv
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    audio_name = audio_path.split("/")[-1].split(".")[0]

    # Create tmp folder
    tmp_out_vis = "outputs/vis"

    os.makedirs(tmp_out_vis, exist_ok=True)

    tmp_out_video = os.path.join(tmp_out_vis, f"{audio_name}.mp4")

    video = cv2.VideoWriter(tmp_out_video, fourcc, fps, (1500, 100))
    for frame in frames:
        video.write(frame)
    video.release()

    # Add audio to video using ffmpeg
    # Create output_video_path dir
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    os.system(
        f"ffmpeg -i {tmp_out_video} -i {audio_path} -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {output_video_path}"
    )
    # os.remove(tmp_out_video)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_output")
    args = parser.parse_args()

    output_dir_name = osp.basename(osp.normpath(args.path_to_output))
    json_files = glob(f"./outputs/{output_dir_name}/submission/*.json")
    output_video = f"./outputs/video/{output_dir_name}/"
    os.system(f'rm -r {output_video}')
    os.makedirs(output_video, exist_ok=True)
    # Generate karaoke video for each json file and audio file
    
    def make_mp4(json_file):
        fname = os.path.basename(json_file).replace(".json", "")
        audio_file = glob(f"./data/*/songs/{fname}.wav")[0]

        if not osp.exists(audio_file):
            audio_file = os.path.join(
                f"./data/{output_dir_name}/songs",
                os.path.basename(json_file).replace(".json", ".mp3"),
            )
        try:
            print(json_file, audio_file)
            output_video_path = os.path.join(
                output_video,
                os.path.basename(json_file).replace(".json", ".mp4"),
            )
            generate_karaoke_video(audio_file, json_file, output_video_path)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print("Error", json_file, audio_file)
            
    for json_file in json_files:
        make_mp4(json_file)