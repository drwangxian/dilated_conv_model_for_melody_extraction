import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
import os
import argparse
from utils.help_str_file import help_str
from general.shaun.shaun import main as general_melody_fn
from vocal.shaun.shaun_vocal import main as vocal_melody_fn
from utils import constants as CONSTS


def parser():

    print('\n\n')

    p = argparse.ArgumentParser(description=str(help_str), formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument('--debug', action='store_true', help='run in debug mode')
    p.add_argument('--melody_type', choices=['general', 'vocal'], default='general', help='melody type, default to general')
    p.add_argument('--mode', choices=['training', 'inference'], default='inference', help='train or inference, default to inference')
    p.add_argument('--inference_dataset', choices=CONSTS.all_allowed_datasets,
                   default='mdb', help='which dataset to run inference on, default to mdb')
    p.add_argument('--test27', action='store_true', help='If present, use partition (66, 15, 27). Otherwise, use (67, 15, 26)')
    p.add_argument('--gpu_idx', type=int, default=0, help='which GPU to use, starting from 0')
    p.add_argument('--ckpt', default=None, help='a checkpoint')
    p.add_argument('--ckpt_prefix', default='d0', help='a checkpoint prefix')
    p.add_argument('--snippet_len', type=int, default=1200)
    p.add_argument('--tb_dir', default='tb_d0', help='directory for tensorboard summary')

    args = p.parse_args()

    return args


def args_processing_fn(args):

    def chk_dot_not_in_path(path):
        assert '../' not in path

    general_melody = args.melody_type == 'general'
    is_inferencing = args.mode == 'inference'
    test26 = not args.test27

    output_args_dict = {}
    output_args_dict['melody_type'] = args.melody_type
    output_args_dict['debug'] = args.debug
    output_args_dict['gpu_idx'] = args.gpu_idx
    output_args_dict['snippet_len'] = args.snippet_len
    output_args_dict['mode'] = args.mode
    output_args_dict['test26'] = test26
    output_args_dict['inference_dataset'] = args.inference_dataset

    # process ckpt_file
    ckpt_file = args.ckpt
    if is_inferencing:

        if ckpt_file is None:

            if general_melody:
                if test26:
                    ckpt_file = os.path.join(os.getcwd(), 'checkpoints/general/shaun/67_15_26', 'd0-39')
                else:
                    ckpt_file = os.path.join(os.getcwd(), 'checkpoints/general/shaun/66_15_27', 'd0-21')
            else:
                ckpt_file = os.path.join(os.getcwd(), 'checkpoints/vocal/shaun/d0-28')
        else:
            chk_dot_not_in_path(ckpt_file)
            ckpt_dir, ckpt_name = os.path.split(ckpt_file)
            assert ckpt_name != ''
            if ckpt_dir == '':
                ckpt_dir = 'ckpts'
            ckpt_file = os.path.join(ckpt_dir, ckpt_name)
            abs_path = os.path.isabs(ckpt_file)
            if not abs_path:
                ckpt_file = os.path.join(os.getcwd(), ckpt_file)

    else:  # is training

        if ckpt_file is not None:
            chk_dot_not_in_path(ckpt_file)
            ckpt_dir, ckpt_name = os.path.split(ckpt_file)
            assert ckpt_name != ''
            if ckpt_dir == '':
                ckpt_dir = 'ckpts'
            ckpt_file = os.path.join(ckpt_dir, ckpt_name)
            abs_path = os.path.isabs(ckpt_file)
            if not abs_path:
                if general_melody:
                    ckpt_file = os.path.join(os.getcwd(), 'general/shaun', ckpt_file)
                else:
                    ckpt_file = os.path.join(os.getcwd(), 'vocal/shaun', ckpt_file)
    output_args_dict['ckpt_file'] = ckpt_file

    # process ckpt_prefix
    ckpt_prefix = args.ckpt_prefix
    if not is_inferencing:
        ckpt_dir, prefix = os.path.split(ckpt_prefix)
        assert prefix != ''
        if ckpt_dir == '':
            ckpt_dir = 'ckpts'
        ckpt_prefix = os.path.join(ckpt_dir, prefix)
        abs_path = os.path.isabs(ckpt_prefix)
        if not abs_path:
            if general_melody:
                ckpt_prefix = os.path.join(os.getcwd(), 'general/shaun', ckpt_prefix)
            else:
                ckpt_prefix = os.path.join(os.getcwd(), 'vocal/shaun', ckpt_prefix)
    else:
        ckpt_prefix = None
    output_args_dict['ckpt_prefix'] = ckpt_prefix

    # process tb_dir
    tb_dir = args.tb_dir
    assert tb_dir != ''
    dir_name, tb_name = os.path.split(tb_dir)
    assert tb_name != ''
    if dir_name == '':
        if general_melody:
            dir_name = 'general/shaun'
        else:
            dir_name = 'vocal/shaun'
    tb_dir = os.path.join(dir_name, tb_name)
    chk_dot_not_in_path(tb_dir)
    abs_path = os.path.isabs(tb_dir)
    if not abs_path:
        tb_dir = os.path.join(os.getcwd(), tb_dir)
    output_args_dict['tb_dir'] = tb_dir


    summaries = []
    line = '*** information summary ***'
    summaries.append(line)
    if is_inferencing:
        line = 'melody type: {}'.format(output_args_dict['melody_type'])
        summaries.append(line)

        line = 'mode: {}'.format(output_args_dict['mode'])
        summaries.append(line)

        line = 'debug: {}'.format(output_args_dict['debug'])
        summaries.append(line)

        line = 'ckpt: {}'.format(output_args_dict['ckpt_file'])
        summaries.append(line)

        line = 'inf. dataset: {}'.format(output_args_dict['inference_dataset'])
        summaries.append(line)

        if general_melody:
            line = 'test27: {}'.format(not output_args_dict['test26'])
            summaries.append(line)

        line = 'GPU idx: {}'.format(output_args_dict['gpu_idx'])
        summaries.append(line)

        line = 'snippet len: {}'.format(output_args_dict['snippet_len'])
        summaries.append(line)

        line = 'tb directory: {}'.format(output_args_dict['tb_dir'])
        summaries.append(line)
    else:
        line = 'melody type: {}'.format(output_args_dict['melody_type'])
        summaries.append(line)

        line = 'mode: {}'.format(output_args_dict['mode'])
        summaries.append(line)

        line = 'debug: {}'.format(output_args_dict['debug'])
        summaries.append(line)

        ckpt = output_args_dict['ckpt_file']
        if ckpt is not None:
            line = 'cont. training from: {}'.format(ckpt)
        else:
            line = 'train from scratch'
        summaries.append(line)

        if general_melody:
            line = 'test27: {}'.format(not output_args_dict['test26'])
            summaries.append(line)

        line = 'GPU idx: {}'.format(output_args_dict['gpu_idx'])
        summaries.append(line)

        line = 'snippet len: {}'.format(output_args_dict['snippet_len'])
        summaries.append(line)

        line = 'ckpt prefix: {}'.format(output_args_dict['ckpt_prefix'])
        summaries.append(line)

        line = 'tb directory: {}'.format(output_args_dict['tb_dir'])
        summaries.append(line)
    line = '*** end summary ***\n\n'
    summaries.append(line)

    for line in summaries:
        logging.info(line)

    return output_args_dict


def main():

    args = parser()

    pargs = args_processing_fn(args)
    general_melody = pargs['melody_type'] == 'general'

    if general_melody:
        general_melody_fn(
            debug=pargs['debug'],
            mode=pargs['mode'],
            inference_dataset=pargs['inference_dataset'],
            test_split_size=26 if pargs['test26'] else 27,
            gpu_idx=pargs['gpu_idx'],
            ckpt_file=pargs['ckpt_file'],
            snippet_len=pargs['snippet_len'],
            ckpt_prefix=pargs['ckpt_prefix'],
            tb_dir=pargs['tb_dir']
        )
    else:
        vocal_melody_fn(
            debug=pargs['debug'],
            mode=pargs['mode'],
            inference_dataset=pargs['inference_dataset'],
            gpu_idx=pargs['gpu_idx'],
            ckpt_file=pargs['ckpt_file'],
            snippet_len=pargs['snippet_len'],
            ckpt_prefix=pargs['ckpt_prefix'],
            tb_dir=pargs['tb_dir']
        )


if __name__ == '__main__':

    main()



