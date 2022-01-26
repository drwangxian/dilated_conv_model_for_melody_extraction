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
- utils: miscellaneous utilities
- vocal: implementations of some models for vocal melody extraction
- predict.py: extract the melodies of any customer audio files with the proposed model
- training_and_inference.py: for training the proposed model and inferencing on known datasets
- data_splits_jiri.json: partition (67, 15, 27) of MedleyDB

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
This is another higher level, packed-up version of the implementation of the proposed model. You can use it for training the proposed model and inferencing on known datasets.
- To train the model,
  - First obtain the MedleyDB dataset, and extract it to a folder. 
  - Create an environment variable named *medleydb* pointing to the above folder.
  - Create a folder named *V1_auxiliary* under *$medleydb*.
  - Move file *data_splits_jiri.json* to *$medleydb/V1_auxiliary*
- To inference on a known dataset except MedleyDB,
  - First obtain the dataset and extract it to a folder.
  - Then create an environment variable pointing to the above folder.
  - For the name of this environment variable, please refer to file *general/shaun/shaun.py* and file *vocal/shaun/shaun_vocal.py* for details.  

```
usage: python training_and_inference.py [-h] [--debug] [--melody_type {general,vocal}] [--mode {training,inference}]
                                        [--inference_dataset {rwc,adc04,mdb_s,mirex05,orchset,mdb,mir1k,wjd}] [--test27] [--gpu_idx GPU_IDX] [--ckpt CKPT]
                                        [--ckpt_prefix CKPT_PREFIX] [--snippet_len SNIPPET_LEN] [--tb_dir TB_DIR]

Instruction on How to Use Melody Extraction

1. Use toggle --debug to specify to run in a debug mode. In this mode, we will use less data for training, validaiton or inference.
   Therefore, use this option to quickly see if the program can run correctly.

2. The program can run in two modes, namely, training and inference. Use --mode to specify the mode.

3. There are two types of melody, namely, general and vocal. Use --melody_type to specify the melody type.

4. We can do inference on multiple datasets. For general melody, the supported datasets are adc04, mdb_s, mirex05, mdb, orchset, and wjd.
   For vocal melody, the supported datasets are adc04, mirex05, mdb, mir1k, and rwc.
   Use --inference_dataset to specify the dataset.

5. The are two possible partitions of the MedleyDB dataset, namely, (67, 15, 26) and (66, 15, 27). 
   Use option --test27 to select (66, 15, 27).
   Otherwise, (67, 15, 26) will be selected by default.

6. Use --gpu_idx to select which GPU to use if you have multiple GPUs. Default to 0.

7. In training mode, we can continue training from an existing checkpoint or train from scratch. In inference mode, a checkpoint is required.
   Use option --ckpt to specify the checkpoint. 
   In inference mode, if a checkpoint is not specified, the checkpoint coming with this paper will be used.

8. If you come across out of memory error, decrease the snippet length to suit your case. 
   Use --snippet_len to do this.

9. In training mode, you can specify a prefix for your checkpoint. Use option --ckpt_prefix to do this. If not supplied, by default 'd0' will be used.
   We will only save one checkpoint. This is the checkpoint that yields the best validation performance. The location of the checkpoint is a folder named
   'ckpts' under the folder where the main program resides. For example, for general model it is ./general/shaun/ckpts.

10. In non-debug mode, the program will throw out an error if the directory for tensorboard already exists.

11. In non-debug mode and training mode, the program will throw out an error if the same checkpoint directory already
    exists. 

12. During training and inference, all the necessary information, such as loss, overall accuracy and voicing threshold, will be saved as 
    tensorboard summary, so that it can be viewed later conveniently. We need a folder for this purpose. This folder is specified with
    --tb_dir. The argument for --tb_dir can have different formats. Next we give some examples about the format of the argument and the corresponding
    absolute path of the folder. In the following, a path that does not start with '/' is a relative path, being relative to the top folder of the 
    current project.
    
            --tb_dir argument | absolute path for general melody | absolute path for vocal melody
            tb_d0               general/shaun/tb_d0                vocal/shaun/tb_d0
            today/tb_d0         general/shaun/today/tb_d0          vocal/shaun/today/tb_d0
            /tmp/tb_d0          /tmp/tb_d0                         /tmp/tb_d0

    We do not accept relative path for the argument of --tb_dir, such as ../tb_d0.
    In non-debug mode, the program will throw out an error if the folder for tensorboard summary already exists. 
    So if you continue training from a previous checkpoint and the previous folder for tensorboard is tb_d0, you should 
    specify a different folder for the continued training, e.g, tb_d1, to avoid the above error.

13. More instructions on how to specify the argument of --ckpt_prefix. In training mode, the best checkpoint is stored 
    in a folder. For continuing training, we also need to read the previous checkpoint from a folder. 
    The argument of --ckpt_prefix has two functions. The directory name of it specify the folder. The basename 
    of it specify the prefix for the checkpoint. Below are some examples for the format of the argument and the 
    corresponding absolute path of the folder.
            --ckpt_prefix argument | absolute path for general melody | absolute path for vocal melody
            d0                       general/shaun/ckpts                vocal/shaun/ckpts
            ckpts_d0/d0              general/shaun/ckpts_d0             vocal/shaun/ckpts_d0 
            /tmp/ckps/d0             /tmp/ckps                          /tmp/ckpts
    The checkpoint will be stored in the above folder and named in a format of 'd0-n', where n is the epoch number starting 
    from 1. Note that we only store the best checkpoint that yields the best validation performance. Therefore, any existing 
    checkpoint under the same folder has the risk of being overwritten. To emphasize this risk, in non-debug and training 
    mode, we will check if the folder already exists. If this is the case, an error will be thrown out. Thus, for 
    continuing training, if the previous checkpoint is stored in ckpts/, for the current training it is better to 
    specify the argument of --ckpt_prefix as ckpts_d1/d1. 
    We do not accept any relative path for the argument.

14. More instructions on how to specify the argument of --ckpt. We need a checkpoint for continuing training and
    inference. It is specified with --ckpt. Below are some examples for the format of the argument of --ckpt and the 
    corresponding absolute path of the checkpoint.
            --ckpt argument | absolute path for general melody | absolute path for vocal melody    
            d0-39             general/shaun/ckpts/d0-39          vocal/shaun/ckpts/d0-39 
            ckpts_d1/d1-39    general/shaun/ckpts_d1/d1_39       vocal/shaun/ckpts_d1/d1_39
            /tmp/ckpts/d1-39  /tmp/ckpts/d1-39                   /tmp/ckpts/d1-39
    The existing checkpoint may have two associated files looking like 'd0-39.data-00000-of-00001' and 'd0-39.index'. In
    this case only specify the argument as d0-39.
    We do not accept any relative path for this argument.   
    
    

optional arguments:
  -h, --help            show this help message and exit
  --debug               run in debug mode
  --melody_type {general,vocal}
                        melody type, default to general
  --mode {training,inference}
                        train or inference, default to inference
  --inference_dataset {rwc,adc04,mdb_s,mirex05,orchset,mdb,mir1k,wjd}
                        which dataset to run inference on, default to mdb
  --test27              If present, use partition (66, 15, 27). Otherwise, use (67, 15, 26)
  --gpu_idx GPU_IDX     which GPU to use, starting from 0
  --ckpt CKPT           a checkpoint
  --ckpt_prefix CKPT_PREFIX
                        a checkpoint prefix
  --snippet_len SNIPPET_LEN
  --tb_dir TB_DIR       directory for tensorboard summary
```
