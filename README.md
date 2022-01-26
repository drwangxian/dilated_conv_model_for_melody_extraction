# Dilated Convolutional Model for Melody Extraction
X. Wang, L. Liu, and Q. Shi, "Dilated convolutional model for melody extraction," submitted to IEEE Signal Processing Letters for peer review, 2022.

This is the accompanying code for the above paper.

### What are included in this respsitory?
- the implementatons of the proposed model as well as some existing models compared with in the paper
- the checkpoints of some trained models
- two higher level, packed-up versions of the implementation of the proposed model

### What are the folders and files in this repository?
- checkpoints: checkpoints of some trained models
- geneal: implementations of some models for general meldoy extraction
- vocal: implementations of some models for vocal melody extraction
- predict.py: extract the melodies of any customer audio files with the proposed model
- training_and_inference.py: for training the proposed model and inferencing on known datasets

### Main Dependencies
- python 3.8.8 
- tensorflow 2.x
- librosa 0.7.2
- SoundFile 0.10.3.post1
- sqlite 3.35.4
- medleydb 1.3.4
- mir-eval 0.6

### How to use predict.py
To use this script, you do not need any datasets. This script is a higher level, packed-up version of the implementation of the proposed model. It uses our trained models to extract geneal or vocal melodies of customer audio files. The melody will be written to a csv file in which the first column are times in seconds and the second column are frequencies in Hertz.
```
usage: python predict.py [-h] [--output_dir [OUTPUT_DIR]] [--melody_type {general,vocal}] [--gpu_idx GPU_IDX] [--test27] input_files [input_files ...]

extract general or vocal melody

positional arguments:
  input_files           one or multiple input audio files, supporting wildcard characters

optional arguments:
  -h, --help            show this help message and exit
  --output_dir [OUTPUT_DIR]
                        output directory
  --melody_type {general,vocal}
                        melody type: general or vocal, defaults to general
  --gpu_idx GPU_IDX     which GPU to use, starting from 0, defaults to 0
  --test27              Relevant only for general melody extraction.
                        Determine which checkpoint to use for general melody extraction.
                        Use flag --test27 to select the checkpoint for partition (66, 15, 27).
                        Otherwise, the checkpoint for partition (67, 15, 26) will be used
```

### How to use training_and_inference.py
