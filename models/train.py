import os
import numpy as np
import torch as th
from torch import optim
from ..utils.EarlyStop import EarlyStoppingCriterion  
from ..utils.parser import parse_args 
from ..utils.dataloader import load_data 
from minibatch import MinibatchIterator
from models import DSRGAT

PATH_TO_DATA = './movie/'

seed = 123
np.random.seed(seed)
th.manual_seed(seed)

def train(args, data):

    adj_tensor, latest_sessions, user_id_map, item_id_map, train_tensor, valid_tensor, test_tensor = data
    args.num_items = len(item_id_map) + 1
    args.num_users = len(user_id_map)

    minibatch = MinibatchIterator(
        adj_tensor, latest_sessions, [train_tensor, valid_tensor, test_tensor],
        batch_size=args.batch_size, max_degree=args.max_degree,
        num_nodes=len(user_id_map), max_length=args.max_length,
        samples_1_2=[args.samples_1, args.samples_2]
    )
    

    model = DSRGAT(args, minibatch.sizes)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    early_stopping = EarlyStoppingCriterion(
        patience=10, delta=0.0, save_path=os.path.join(args.ckpt_dir, 'best_model.pth'), verbose=True
    )

    for epoch in range(args.epochs):
        model.train()
        minibatch.shuffle()
        print(f'Epoch: {epoch + 1}')

        for step, batch in enumerate(minibatch):
            optimizer.zero_grad()
            loss, recall, ndcg, point_count = model(batch)
            loss.backward()
            optimizer.step()

        val_loss, val_recall, val_ndcg = evaluate(model, minibatch, phase='val')
        print(f"Validation at epoch {epoch + 1}: val_loss={val_loss:.5f}, val_recall@20={val_recall:.5f}, val_ndcg={val_ndcg:.5f}")

        early_stopping(val_recall, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Load the best model for final test evaluation
    model.load_state_dict(th.load(os.path.join(args.ckpt_dir, 'best_model.pth')))
    print('Best model loaded for final evaluation on the test set.')
    test_loss, test_recall, test_ndcg = evaluate(model, minibatch, phase='test')
    print(f"Test results: Loss={test_loss:.5f}, Recall@20={test_recall:.5f}, NDCG={test_ndcg:.5f}")

def evaluate(model, minibatch, phase='val'):
    model.eval()
    total_loss, total_recall, total_ndcg, total_points = 0, 0, 0, 0
    
    with th.no_grad():
        for batch in minibatch.get_val_batches(phase):
            loss, recall, ndcg, point_count = model(batch)
            total_loss += loss.item() * point_count
            total_recall += recall * point_count
            total_ndcg += ndcg * point_count
            total_points += point_count

    avg_loss = total_loss / total_points
    avg_recall = total_recall / total_points
    avg_ndcg = total_ndcg / total_points
    return avg_loss, avg_recall, avg_ndcg

def main():
    args = parse_args()
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    
    print('Loading data...')
    data = load_data(PATH_TO_DATA)
    print("Data loaded successfully!")
    
    train(args, data)

if __name__ == '__main__':
    main()

