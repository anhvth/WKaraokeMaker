# WKaraokeMaker
## Description

This work earns first place solution of [ZAC2022](https://challenge.zalo.ai/portal/lyric-alignment/final-leaderboard), LyricAlignment Track.

## What is Karaoke Maker?

Karaoke Maker is a task that predicts the lyrics and melody of the given music audio. 
It is a task that can be used in various fields such as music production and karaoke you name it.

## Installation
* ffmpeg # for audio/video processing
* python >= 3.8
* pytorch torchaudio

```bash
    conda create -n kmaker python 3.8
    # Install pytorch https://pytorch.org/get-started/locally/
    pip install git+https://github.com/openai/whisper.git 
    pip install -r requirements
    pip install -e ./
```


## Example karaok video
```
    https://github.com/anhvth/WKaraokeMaker/blob/main/ee6eb1a3b2bdf727.mp4
```
