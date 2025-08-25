import os, yaml, argparse
from time import sleep
import numpy as np
from src.generator import Generator
from src.processor import Processor


def main():
    # Loading parameters
    # gc.set_debug(gc.DEBUG_LEAK)
    parser = init_parser()
    args = parser.parse_args()
    args = update_parameters(parser, args)  # cmd > yaml > default

    # Waiting to run
    sleep(args.delay_hours * 3600)

    # folds = [1, 2, 3, 4, 5]  # 你可以根据需要更改这个范围
    # seeds = [42, 3072, 1, 2, 3, 4, 5, 6, 7, 8, 112, 232, 401, 314, 315, 211, 101, 1225, 909]
    # seeds = [1] * 10
    cls_pow = [0.25]
    # metrics = ['AIM', 'LEM', 'LCM', 'MIX']
    results = np.zeros(len(cls_pow))
    for index, pow in enumerate(cls_pow):
        # args.metric = metric  # 修改 metric 值
        args.model_args['cls_pow'] = pow  # 修改 fold 值
        # args.test_feeder_args['fold'] = fold  # 修改 fold 值
        print(f"Running with pow={pow}...")  # 输出当前fold的值，方便追踪

        # Processing
        if args.generate_data:
            g = Generator(args)
            g.start()

        elif args.extract:
            p = Processor(args)
            p.extract()

        else:
            p = Processor(args)
            acc = p.start()
            results[index] = acc
        # Add a short sleep to avoid potential issues with resource contention (if needed)
        sleep(1)

    results = results * 100
    # 输出结果
    print("Results (in percentage):", np.round(results, 2))
    print("Mean: {:.2f}%".format(np.mean(results)))
    print("Standard Deviation: {:.2f}%".format(np.std(results)))


def init_parser():
    parser = argparse.ArgumentParser(description='Method for Skeleton-based Action Recognition')

    # Setting
    parser.add_argument('--config', '-c', default='./configs/ntu_mutual/xview.yaml',
                        help='path to the config file')
    parser.add_argument('--gpus', '-g', type=int, nargs='+', default=[0], help='Using GPUs')
    parser.add_argument('--seed', '-s', type=int, default=1, help='Random seed')
    parser.add_argument('--pretrained_path', '-pp', type=str, default='', help='Path to pretrained models')
    parser.add_argument('--work_dir', '-w', type=str, default='./', help='Work dir')
    parser.add_argument('--no_progress_bar', '-np', default=False, action='store_true', help='Do not show progress bar')
    parser.add_argument('--delay_hours', '-dh', type=float, default=0, help='Delay to run')

    # Processing
    parser.add_argument('--debug', '-db', default=False, action='store_true', help='Debug')
    parser.add_argument('--resume', '-r', default=False, action='store_true', help='Resume from checkpoint')
    parser.add_argument('--evaluate', '-e', default=False, action='store_true', help='Evaluate')
    parser.add_argument('--extract', '-ex', default=False, action='store_true', help='Extract')
    parser.add_argument('--generate_data', '-gd', default=False, action='store_true', help='Generate skeleton data')

    # Dataloader
    parser.add_argument('--dataset', '-d', type=str, default='', help='Select dataset')
    parser.add_argument('--dataset_args', default=dict(), help='Args for creating dataset')

    # Model
    parser.add_argument('--model_type', '-mt', type=str, default='', help='Args for creating model')
    parser.add_argument('--model_args', default=dict(), help='Args for creating model')

    # Optimizer
    parser.add_argument('--optimizer', '-o', type=str, default='', help='Initial optimizer')
    parser.add_argument('--optimizer_args', default=dict(), help='Args for optimizer')

    # LR_Scheduler
    parser.add_argument('--lr_scheduler', '-ls', type=str, default='', help='Initial learning rate scheduler')
    parser.add_argument('--scheduler_args', default=dict(), help='Args for scheduler')

    return parser


def update_parameters(parser, args):
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            try:
                yaml_arg = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_arg = yaml.load(f)
            default_arg = vars(args)
            for k in yaml_arg.keys():
                if k not in default_arg.keys():
                    raise ValueError('Do NOT exist this parameter {}'.format(k))
            parser.set_defaults(**yaml_arg)
    else:
        raise ValueError('Do NOT exist this file in \'configs\' folder: {}!'.format(args.config))
    return parser.parse_args()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
