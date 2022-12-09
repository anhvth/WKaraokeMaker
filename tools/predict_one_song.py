import os
import os.path as osp
from glob import glob

import cv2
import mmcv
import numpy as np
import torch
import torchaudio
from kmaker.dataloader import *
# from moviepy.editor import AudioFileClip, VideoFileClip
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from tools.test_submit import (Segment, convert_result_to_competion_format,
                               load_eval_model, preproc)

FONT = '/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf'
assert osp.exists(FONT), f'Font {FONT} does not exist'
def torch_load_audio(audio_path, sr=16000):
    """
    Load audio using torchaudio
    """
    audio = torchaudio.load(audio_path)[0]
    return audio.numpy()


class UTF8TextWriter:
    def __init__(self, color=(255, 0, 0)):
        self.cv2_img_add_text = self.init_parameters(self.cv2_img_add_text, font=FONT, text_size=24, text_rgb_color=color)
        
    def __call__(self, img, text, left_corner, **option):
        return self.cv2_img_add_text(img, text, left_corner, **option)



    @staticmethod
    def init_parameters(fun, **init_dict):
        """
        help you to set the parameters in one's habits
        """
        def job(*args, **option):
            option.update(init_dict)
            return fun(*args, **option)
        return job

    @staticmethod
    def cv2_img_add_text(img, text, left_corner, text_rgb_color=(255, 0, 0), text_size=24, font='mingliu.ttc', **option):
        """
        USAGE:
            cv2_img_add_text(img, '中文', (0, 0), text_rgb_color=(0, 255, 0), text_size=12, font='mingliu.ttc')
        """
        pil_img = img
        if isinstance(pil_img, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font_text = ImageFont.truetype(font=font, size=text_size, encoding=option.get('encoding', 'utf-8'))
        draw.text(left_corner, text, text_rgb_color, font=font_text)
        cv2_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
        if option.get('replace'):
            img[:] = cv2_img[:]
            return None
        return cv2_img
    
def generate_karaoke_video(audio_path, lyrics_path, output_video_path, fps=30):
    """
    Generate karaoke video from audio and lyrics
    
    """
    # Load audio and lyrics
    # audio = torch_load_audio(audio_path, sr=16000)
    utf8_text_writer = UTF8TextWriter()
    utf8_text_writer_green = UTF8TextWriter(color=(0, 255, 0))

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
    height, width = 300, 1280
    frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)

    for i, line in enumerate(lyrics):
        start_line_frame = int((line["s"] - start_time) * fps)
        end_line_frame = int((line["e"] - start_time) * fps)

        text_line = " ".join([word["d"] for word in line["l"]])
        first_frame_line = frames[start_line_frame].copy() * 0
        # Puttext to first frame using opencv
        # text_line = convert_vietnamese_text_to_english_text(text_line)
        # Put bold text to first frame in the center

        # cv2.putText(
        #     first_frame_line,
        #     text_line,
        #     (int(width / 5), int(height / 2)),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (255, 255, 255),
        #     2,
        #     cv2.LINE_AA,
        # )
        first_frame_line = utf8_text_writer(first_frame_line, text_line, (int(width / 5), int(height / 2)), text_rgb_color=(255, 255, 255), text_size=1)



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

            # Puttext to first frame of this word 
            middle_word_frame = frames[
                start_word_frame + (end_word_frame - start_word_frame)// 2
            ].copy()
            middle_word_frame = utf8_text_writer_green(middle_word_frame, current_text_line, (int(width / 5), int(height / 2)), text_rgb_color=(0, 255, 0), text_size=1)


            frames[start_word_frame:end_word_frame] = middle_word_frame


    # Generate video from frames using opencv
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    audio_name = audio_path.split("/")[-1].split(".")[0]

    # Create tmp folder
    tmp_out_vis = "outputs/vis"

    os.makedirs(tmp_out_vis, exist_ok=True)

    tmp_out_video = os.path.join(tmp_out_vis, f"{audio_name}.mp4")

    video = cv2.VideoWriter(tmp_out_video, fourcc, fps, (width, height))
    for frame in frames:
        video.write(frame)
    video.release()
    # Add audio to video using ffmpeg
    # Create output_video_path dir
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    os.system(
        f"ffmpeg -i {tmp_out_video} -i {audio_path} -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {output_video_path}"
    )
    os.remove(tmp_out_video)

def make_mp4(json_file, audio_file, output_video_path):
    """
    Make mp4 from json and audio
    """
    if not osp.exists(audio_file):
        audio_file = os.path.join(
            f"./data/{output_dir_name}/songs",
            os.path.basename(json_file).replace(".json", ".mp3"),
        )
    try:
        print(json_file, audio_file)
        generate_karaoke_video(audio_file, json_file, output_video_path)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print("Error", json_file, audio_file)

def get_audio_file(json_file, data_dir="data"):
    """
        From json file, get audio file
    """
    fname = os.path.basename(json_file).replace(".json", "")
    return glob(f"{data_dir}/*/songs/{fname}.*")[0]
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("path_to_output")
    parser.add_argument('sample_json')
    parser.add_argument('--no-sot', dest='sot', action='store_false')
    args = parser.parse_args()

    
    # for json_file in json_files:
    json_file = args.sample_json
    
    eval_model = load_eval_model('lightning_logs/base_detection_no_ckpt_1k/08/ckpts/epoch=116_val_loss_giou=0.0000.ckpt', sot=True)
    collate_fn = collate_fn_with_sot if args.sot else collate_fn_without_sot
    item, batch = preproc(json_file, collate_fn)
    with torch.inference_mode():
        outputs = eval_model.forward_both(
                    batch['inputs'],
                    labels=batch['labels'],
                    ctc_labels=batch['w2v_labels'],
                )
        bboxes = outputs['bbox_pred'][batch['dec_pos']]
    # import box_cxcywh_to_xyxy
    from kmaker.segment_utils import box_cxcywh_to_xyxy

    xyxy = box_cxcywh_to_xyxy(bboxes)[:,[0,2]]
    words = [_[0] for _ in item.words]
    
    results = []
    xyxy = xyxy.clip(0, 1)
    for (x1,x2), word in zip((xyxy*30).tolist(), words):
        results.append((word, x1, x2))
    results = [Segment(*result, 1) for result in results]
    results = convert_result_to_competion_format(results, json_file, 1000)    
    
    fname = os.path.basename(json_file).replace(".json", "")
    audio_file = get_audio_file(json_file)
    output_video = '/tmp/karaoke'
    output_video_path = os.path.join(
        output_video,
        os.path.basename(json_file).replace(".json", ".mp4"),
    )
    make_mp4(json_file, audio_file, output_video_path)
    print('-> {}'.format(osp.abspath(output_video_path)))
