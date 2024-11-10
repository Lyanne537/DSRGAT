import os 
import sys
import numpy as np
import torch as th
from ..utils.parser import parse_args  
from ..utils.dataloader import load_data 
from minibatch import MinibatchIterator
from models import DSRGAT

PATH_TO_DATA = './movie/'

seed = 123
np.random.seed(seed)
th.manual_seed(seed)

def evaluate(model, minibatch, phase='test'):
    model.eval()
    total_loss, total_recall, total_ndcg, total_points = 0, 0, 0, 0
    input_str = []
    
    with th.no_grad():
        for batch in minibatch.get_val_batches(phase):
            loss, recall, ndcg, point_count = model(batch)
            x = batch['input_x'].view(-1).tolist()
            x_str = '_'.join([str(v) for v in x if v != 0])
            input_str.append(x_str)
            total_loss += loss.item() * point_count
            total_recall += recall * point_count
            total_ndcg += ndcg * point_count
            total_points += point_count

    avg_loss = total_loss / total_points
    avg_recall = total_recall / total_points
    avg_ndcg = total_ndcg / total_points
    return avg_loss, avg_recall, avg_ndcg, total_recall, total_ndcg, input_str

def test(args, data):
    adj_tensor, latest_sessions, user_id_map, item_id_map, train_tensor, valid_tensor, test_tensor = data
    args.num_items = len(item_id_map) + 1
    args.num_users = len(user_id_map)
    args.batch_size = 1 
    
    minibatch = MinibatchIterator(
        adj_tensor, latest_sessions, [train_tensor, valid_tensor, test_tensor],
        batch_size=args.batch_size, max_degree=args.max_degree, 
        num_nodes=len(user_id_map), max_length=args.max_length,
        samples_1_2=[args.samples_1, args.samples_2], training=False
    )

    model = DSRGAT(args, minibatch.sizes)
    checkpoint_path = os.path.join(args.ckpt_dir, 'best_model.pth')

    if os.path.exists(checkpoint_path):
        model.load_state_dict(th.load(checkpoint_path))
        print(f'Model restored from {checkpoint_path}!')
    else:
        print(f'Failed to restore model from {args.ckpt_dir}')
        sys.exit(0)

    ret = evaluate(model, minibatch, phase="test")
    print("Test results (batch_size=1):",
          f"\tloss={ret[0]:.5f}",
          f"\trecall@20={ret[1]:.5f}",
          f"\tndcg={ret[2]:.5f}")

    with open('metric_dist.txt', 'w') as f:
        for idx in range(len(ret[-1])):
            f.write(f"{ret[-1][idx]}\t{ret[-3][idx]:.5f}\t{ret[-2][idx]:.5f}\n")

def main():
    args = parse_args()  
    print('Loading data...')
    data = load_data(PATH_TO_DATA)
    print("Data loaded successfully!")
    
    test(args, data)

if __name__ == '__main__':
    main()
