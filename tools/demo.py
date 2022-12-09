"""
Example streamlit app for the demo,
    Input: Audio file and lyrics
    Output: Karaoke video with lyrics visualization with streamlit
    Example command: streamlit run tools/demo.py
"""
# Import libraries
import os
# try import streamlit if the module does not exist then install it with os.system
try:
    import streamlit as st
except ModuleNotFoundError:
    os.system("pip install streamlit")
    import streamlit as st
from glob import glob
# import generate_karaoke_video
from tools.make_karaoke_video import generate_karaoke_video

if __name__ == "__main__":
    # Set up streamlit
    st.title("Karaoke Video Demo")
    st.write("This is a demo for the karaoke video generation")
    st.write("Please upload an audio file and lyrics file")
    st.write("The audio file should be in wav format")
    st.write("The lyrics file should be in txt format")
    st.write("The lyrics file should be in the format of")
    st.write("start_time end_time text")
    st.write("The start_time and end_time should be in seconds")
    st.write("The text should be the lyrics of the audio file")
    st.write("Please note that the lyrics text should be in Vietnamese")
    st.write("The output video will be generated in the output folder")

    lyrics_file = st.file_uploader("Upload Lyrics", type=["json"])
    if lyrics_file is not None:
        # Save lyrics to data folder
        lyrics_path = os.path.join("data", "lyrics.json")
        file_name = lyrics_file.name
        
        with open(lyrics_path, "wb") as f:
            f.write(lyrics_file.getbuffer())

    # Generate karaoke video
    if lyrics_file is not None:
        query = './data/*/songs/' + file_name.replace('.json', '.*')
        print(query)
        audio_path = glob(query)[0]
        output_video_path = os.path.join("output", "output.mp4")
        # remove output_video_path
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        generate_karaoke_video(audio_path, lyrics_path, output_video_path)

        # Convert output_video_path to support streamlit video format (avi) using ffmpeg

        tmp_output_video_path = os.path.join("output", "output.mov")
        cmd = f"/usr/bin/ffmpeg -y -i {output_video_path} -vcodec libx264 {tmp_output_video_path}"
        st.write(cmd)
        os.system(cmd)
        
        output_video_path = tmp_output_video_path
        video = open(output_video_path, "rb").read()
        st.video(video)