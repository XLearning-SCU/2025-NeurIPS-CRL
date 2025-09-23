import argparse


def str2bool(value):
    """将命令行参数中的字符串转换为布尔值"""
    if value.lower() in ('true', 't', '1', 'yes', 'y'):
        return True
    elif value.lower() in ('false', 'f', '0', 'no', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid truth value {value}")


def get_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--dataset', default='change_object', type=str, help='Eval dataset')
    parser.add_argument('--all_text', type=int, default=50)
    parser.add_argument('--combiner_mode', type=str, default='image_only')
    parser.add_argument('--pretrained', type=str2bool, default=True)
    args = parser.parse_args()

    return args

