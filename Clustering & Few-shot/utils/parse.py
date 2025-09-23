import argparse


def get_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=1, help='gpu id')
    parser.add_argument('--data_path', type=str, default="/xlearning/honglin/datasets/multi-label", help='data path')
    parser.add_argument('--dataset', type=str, default="clevr4", help='clevr4, cards, stanford-cars')
    parser.add_argument("--seed", type=int, default=0, help='random seed')
    parser.add_argument('--criterion', type=str, default="color", help='clustering criterion')
    parser.add_argument('--all_text', type=int, default=500, help='number of generated text')
    parser.add_argument('--shots', type=int, default=10, help='number of shots')
    args = parser.parse_args()
    return args
