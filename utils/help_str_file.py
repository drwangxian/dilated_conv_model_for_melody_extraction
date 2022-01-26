

help_str = '''\nInstruction on How to Use Melody Extraction
\n1. Use toggle --debug to specify to run in a debug mode. In this mode, we will use less data for training, validaiton or inference.
   Therefore, use this option to quickly see if the program can run correctly.
\n2. The program can run in two modes, namely, training and inference. Use --mode to specify the mode.
\n3. There are two types of melody, namely, general and vocal. Use --melody_type to specify the melody type.
\n4. We can do inference on multiple datasets. For general melody, the supported datasets are adc04, mdb_s, mirex05, mdb, orchset, and wjd.
   For vocal melody, the supported datasets are adc04, mirex05, mdb, mir1k, and rwc.
   Use --inference_dataset to specify the dataset.
\n5. The are two possible partitions of the MedleyDB dataset, namely, (67, 15, 26) and (66, 15, 27). 
   Use option --test27 to select (66, 15, 27).
   Otherwise, (67, 15, 26) will be selected by default.
\n6. Use --gpu_idx to select which GPU to use if you have multiple GPUs. Defaults to 0.
\n7. In training mode, we can continue training from an existing checkpoint or train from scratch. In inference mode, a checkpoint is required.
   Use option --ckpt to specify the checkpoint. 
   In inference mode, if a checkpoint is not specified, the checkpoint coming with this paper will be used.
\n8. If you come across out of memory error, decrease the snippet length to suit your case. 
   Use --snippet_len to do this.
\n9. In training mode, you can specify a prefix for your checkpoint. Use option --ckpt_prefix to do this. If not supplied, by default 'd0' will be used.
   We will only save one checkpoint. This is the checkpoint that yields the best validation performance. The location of the checkpoint is a folder named
   'ckpts' under the folder where the main program resides. For example, for general model it is ./general/shaun/ckpts.
\n10. In non-debug mode, the program will throw out an error if the directory for tensorboard already exists.
\n11. In non-debug mode and training mode, the program will throw out an error if the same checkpoint directory already
    exists. 
\n12. During training and inference, all the necessary information, such as loss, overall accuracy and voicing threshold, will be saved as 
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
\n13. More instructions on how to specify the argument of --ckpt_prefix. In training mode, the best checkpoint is stored 
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
\n14. More instructions on how to specify the argument of --ckpt. We need a checkpoint for continuing training and
    inference. It is specified with --ckpt. Below are some examples for the format of the argument of --ckpt and the 
    corresponding absolute path of the checkpoint.
            --ckpt argument | absolute path for general melody | absolute path for vocal melody    
            d0-39             general/shaun/ckpts/d0-39          vocal/shaun/ckpts/d0-39 
            ckpts_d1/d1-39    general/shaun/ckpts_d1/d1_39       vocal/shaun/ckpts_d1/d1_39
            /tmp/ckpts/d1-39  /tmp/ckpts/d1-39                   /tmp/ckpts/d1-39
    The existing checkpoint may have two associated files looking like 'd0-39.data-00000-of-00001' and 'd0-39.index'. In
    this case only specify the argument as d0-39.
    We do not accept any relative path for this argument.   
    
    



'''

if __name__ == '__main__':

    s = str(help_str)
    print(s)


