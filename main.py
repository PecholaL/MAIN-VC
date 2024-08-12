"""train MAIN-VC
"""

import yaml
from argparse import ArgumentParser
from solver import Solver

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-config", "-c", default="config.yaml")
    parser.add_argument("-data_dir", "-d")
    parser.add_argument("-train_set", default="train")
    parser.add_argument("-train_index_file")
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--load_opt", action="store_true")
    parser.add_argument("-store_model_path")
    parser.add_argument("-load_model_path")
    parser.add_argument("-log_dir")
    parser.add_argument("-tag", "-t", default="train_log")
    parser.add_argument("-iters", default=5, type=int)
    parser.add_argument("-summary_steps", default=100, type=int)
    parser.add_argument("-save_steps", default=5000, type=int)

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    solver = Solver(config=config, args=args)

    if args.iters > 0:
        solver.train(n_iterations=args.iters)

    complete = input("[MAIN-VC]Training Completed! 🍹")
