# WKaraokeMaker
This is a project to generate karaoke videos from a song and a voice recording. It is based on the [Whisper](https://github.com/openai/whisper.git) project.

## Installation
### Requirements
* ffmpeg
* python 3.8
* pytorch
* [whisper](https://git)
* [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/new-project.html)
* [captum](https://captum.ai/)
* [pydub](

```bash
    conda create -n kmaker python 3.8
    # Install pytorch https://pytorch.org/get-started/locally/
    pip install git+https://github.com/openai/whisper.git 
    pip install -r requirements
    pip install -e ./
```
