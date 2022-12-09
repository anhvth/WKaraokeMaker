# Author: AnhVo
# Date: 2022-12-12

import argparse, os, os.path as osp, glob
from kmaker.video_writer import make_karaoke_video


def get_audio_file(json_file, song_dir):
    """
        From json file, get audio file
    """
    fname = os.path.basename(json_file).replace(".json", "")
    return glob.glob(osp.join(song_dir, fname+'.*'))[0] # get first file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert prediction from submission/*.json to video")
    parser.add_argument("predict_dir", help="Path to submission directory")
    parser.add_argument("song_dir", help="Path to song directory")
    parser.add_argument("output_dir", help="Path to output karaoke video directory")
    args = parser.parse_args()

    # output_dir_name = osp.basename(osp.normpath(args.path_to_output))
    json_files = glob.glob(args.predict_dir + "/*.json")
    assert len(json_files) > 0, "No json file found in {}".format(args.predict_dir)
            
    for json_file in json_files:
        audio_file = get_audio_file(json_file, args.song_dir)
        output_video_path = os.path.join(
            args.output_dir,
            os.path.basename(json_file).replace(".json", ".mp4"),
        )
        make_karaoke_video(json_file, audio_file, output_video_path)
        print('-> {}'.format(osp.abspath(output_video_path)))
