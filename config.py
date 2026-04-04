import argparse



parser = argparse.ArgumentParser(description='ABC', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

## dataset and model
parser.add_argument('--dataset', type=str, default='office', help='office,officehome,visda')
parser.add_argument('--source', type=int, default=0)
parser.add_argument('--target', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--bottle_neck_dim', type=int, default=256, help='bottle_neck_dim')
parser.add_argument("--target_type", default="OPDA", type=str)

## training parameters
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_scale', type=float, default=0.1) 
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--total_epoch', type=int, default=10, help='total epochs')
parser.add_argument('--thresh', type=float, default=None)

## BNC parameters
parser.add_argument('--max_k', type=int, default=100)
parser.add_argument('--covariance_prior', type=float, default=0.001)
parser.add_argument('--alpha', type=float, default=1)

## other parameters
parser.add_argument('--balance', type=float, default=0.01)
parser.add_argument('--lambdav', type=float, default=0.)
parser.add_argument('--KK', type=int, default=5)

args = parser.parse_args()  


