import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=777, help="random seed")
parser.add_argument("--device", type=str, default="cuda:0",
                    help="specify cuda devices")

# hyper-parameters
parser.add_argument("--dataset", type=str, default="politifact",
                    help="[politifact, gossipcop, MMCOVID]")

parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
parser.add_argument("--weight_decay", type=float,
                    default=0.01, help="weight decay")
parser.add_argument("--nhid", type=int, default=128,
                    help="hidden layer dimension for GNN")
parser.add_argument("--feature", type=str, default="glove",
                    help="feature type, [hand, tfidf, glove, bert]")
parser.add_argument("--base_model", type=str, default="gat",
                    help="model type, [gcn, gat, sage]")
parser.add_argument("--save_dir", type=str, default="attack_log",
                    help="the attack logging directory")

parser.add_argument("--mlp_hidden", type=int, default=64,
                    help="hidden layer dimension for MLP in Q net")
parser.add_argument("--max_lv", type=int, default=2,
                    help="max pooling layers for Q net")
parser.add_argument("--latent_dim", type=int, default=128,
                    help="hidden layer dimension for Q net")

parser.add_argument("--burn_in", type=int, default=1, help="burn in steps")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--mem_size", type=int, default=1000000,
                    help="replay memory cell size")
parser.add_argument("--nhop", type=int, default=2, help="number of hops")
parser.add_argument("--num_steps", type=int, default=1001,
                    help="agent training step")
parser.add_argument("--reward_type", type=str,
                    default="nll", help="nll or binary")
parser.add_argument("--num_mod", type=int, default=5, help="budget")
parser.add_argument("--phase", type=str, default="train",
                    help="model phase, train or test")
parser.add_argument("--bilin_q", type=bool, default=True,
                    help="whether using bilinear Q function")

parser.add_argument('--ptb_rate', type=float,
                    default=0.05,  help='pertubation rate')

args = parser.parse_args()
