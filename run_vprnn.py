"""
Executing VP-RNN
------
This file orchestrates various use cases for the Variational Poisson-RNN (VP-RNN) as introduced in Section 3.2 of the paper.

In particular, this script allows to (1) train the model, (2) generate predictions, and (3) compute inventory decisions using
the queuing model described in Section 3.1 on an arbitrary number of stations.
"""

from __future__ import print_function
import argparse
import tqdm
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import pyro
import datetime
from workalendar.usa.new_york import NewYork
from pyro.infer import SVI, Trace_ELBO, Predictive
from src.algos.vprnn import VPRNN
import src.algos.inventory_decision_hourly as idh
import src.algos.inventory_decision_quarterly as idq
from src.misc.utils import get_performance_metrics, Trace_ELBO_Wrapper, read_and_preprocess_data, get_results, get_inventory_decision_evaluation_results


parser = argparse.ArgumentParser(description='Full pipeline example')
# RNN parameters
parser.add_argument('--epochs', type=int, default=50000, metavar='N',
                    help='number of epochs to train (default: 50k)')
parser.add_argument('--no-cuda', type=bool, default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--patience', type=int, default=1000, metavar='N',
                    help='how many epochs without improvement to stop training')
parser.add_argument('--no-train', type=bool, default=False,
                    help='disables training process')
parser.add_argument('--no-predict', type=bool, default=False,
                    help='collects pre-computed prediction')
parser.add_argument('--directory', type=str, default='saved_files',
                    help='defines directory where to save files')

# Data parameters
parser.add_argument('--stations', default=[426], nargs='+',
                    help='list of station IDs on which to run pipeline')
parser.add_argument('--interval', type=int, default=60, metavar='S',
                    help='defines temporal aggregation (defaul 60min)')

# Queuing model parameters
parser.add_argument('--no-decision', default=False, action='store_true',
                    help='disables decision model')
parser.add_argument('--benchmark', action='store_true',
                    help='enables benchmark decision model')

# Parse and preprocess input arguments
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.decision = not args.no_decision
args.train = not args.no_train
args.predict = not args.no_predict
args.interval = args.interval
args.directory = args.directory + f"/{args.interval}min"
if args.interval in [15, 30]:
    args.file_interval = str(args.interval) + 'min'
if args.interval == 60:
    args.file_interval = 'hourly'

# Fix random seed for reproducibility     
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

if args.stations[0] == 'all':
    args.stations = [128, 151, 168, 229, 285, 293, 327, 358, 359, 368, 
                    387, 402, 426, 405, 435, 445, 446, 453, 462, 482, 
                    491, 497, 499, 504, 514, 519, 3263, 3435, 3641, 3711
                    ]

# define test dates
start_date = datetime.date(2018,11,1)
end_date = datetime.date(2018,12,31)
date_list = []
for n in range((end_date - start_date).days + 1):
    dt = start_date + datetime.timedelta(days=n)
    date_list.append(dt)

# loop over selected stations
for station in args.stations:
    try:
        # Load dataset
        df_station, X_train, X_valid, X_test, y_train, y_valid, y_test, X_tensor, y_tensor = \
                 read_and_preprocess_data(demand_path=f"data/demand_rate/{str(args.interval)}min/{str(station)}_{args.file_interval}RatesByDay_2018.csv",
                                        weather_path=f"data/raw/weather2018_{args.interval}min.csv", station_229=False, interval=args.interval)
        labels = ['return', 'pickup']
        # If train==True, start training loop through ELBO maximization (Section 3.2)
        if args.train:
            for i, label in enumerate(labels):
                print(f"\n Training started for St. {station} ({label}), with patience={args.patience}")
                # select return/pickup sequence
                y_train_i, y_valid_i, y_test_i, y_tensor_i = y_train[:,i][:,None], y_valid[:,i][:,None], y_test[:,i][:,None], y_tensor[:,i][:,None]
                X_train_i, X_valid_i, X_test_i, X_tensor_i = X_train, X_valid, X_test, X_tensor
                # train process
                vprnn = VPRNN(input_dim=32, output_dim=1, p_model_dim=128, p_model_layers=1, 
                                q_model_dim=128, q_model_layers=1).to(device)
                svi = SVI(vprnn.model, vprnn.guide, pyro.optim.RMSprop({"lr": 0.001}), Trace_ELBO(num_particles=1))

                train_losses = []
                valid_losses = []
                pyro.clear_param_store()
                epochs = tqdm.trange(args.epochs)
                best_loss = np.inf
                patience = args.patience
                patience_counter = 0

                for epoch in epochs:
                    vprnn.train()
                    loss = 0
                    b = 0
                    X_train_i, y_train_i = X_train_i.to(device), y_train_i.to(device)
                    X_valid_i, y_valid_i = X_valid_i.to(device), y_valid_i.to(device)
                    torch.nn.utils.clip_grad_norm_(vprnn.parameters(),5.)
                    loss += svi.step(X_train_i, y_train_i) / len(X_train_i)
                    vprnn.eval()
                    valid_loss = svi.evaluate_loss(X_valid_i, y_valid_i) / len(X_valid_i)
                    epochs.set_description("Train ELBO: {:.3f} - Test ELBO: {:.3f} - Patience: {:.0f}".format(loss, valid_loss, patience_counter))
                    if valid_loss <= best_loss:
                        torch.save(vprnn.state_dict(), f"{args.directory}/trained_models/rnn/so_st{station}_{label}")
                        best_loss = valid_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"\n Training finished for St. {station} ({label}), loss converged!")
                            break
                    train_losses.append(float(loss))
                    valid_losses.append(float(valid_loss))
                    b += 1
                    del loss
                    del valid_loss
        # If predict==True, generate predictions for test data
        if args.predict:
            for i, label in enumerate(labels):
                y_train_i, y_valid_i, y_test_i, y_tensor_i = y_train[:,i][:,None], y_valid[:,i][:,None], y_test[:,i][:,None], y_tensor[:,i][:,None]
                X_train_i, X_valid_i, X_test_i, X_tensor_i = X_train, X_valid, X_test, X_tensor
                # create model instance and load optimal pre-trained parameters
                vprnn = VPRNN(input_dim=32, output_dim=1, p_model_dim=128, p_model_layers=1, 
                        q_model_dim=128, q_model_layers=1)
                vprnn.load_state_dict(torch.load(f"{args.directory}/trained_models/rnn/so_st{station}_{label}", map_location=device))	

                get_results(model=vprnn, X=X_tensor_i, y=y_tensor_i, station=station, results_path=args.directory, interval=args.interval,
                            model_type="so_rnn", labels=[label], write_mode="w" if i==0 else "a")


############################################################
################ decision pipeline from here ###############
############################################################

        # if decision==True, compute inventory decisions through queuing model (Section 3.1)
        if args.decision:
            # if interval==60min, use hourly predictions as inputs
            if args.interval == 60:
                idh.get_rnn_inventory_decisions(station_id=station, date_list=date_list, hour_range=range(0,24), model_type=['so_rnn',], data_dir='data', prediction_dir=f'{args.directory}/predicted_demand/', result_dir=args.directory)
                if args.benchmark:
                    idh.get_rnn_inventory_decisions(station_id=station, date_list=date_list, hour_range=range(0,24), model_type=['poisson_rnn', 'lr'], data_dir='data', prediction_dir=f'{args.directory}/predicted_demand/', result_dir=args.directory)
                    idh.get_benchmark_inventory_decisions(station, date_list, hour_range=range(0,24), data_dir='data', result_dir=args.directory)
                print(f'Station {station} decision calculation finished!')
                # Evaluation  
                idh.get_inventory_decision_evaluation_results(station, date_list, hour_range=range(0,24), model_type=['so_rnn',], flag_benchmark=args.benchmark, data_dir='data', result_dir=args.directory)
            else:
            # interval==15min or 30min
                idq.get_rnn_inventory_decisions(station_id=station, date_list=date_list, hour_range=range(0,24), quarter=args.interval, model_type=['so_rnn',], data_dir='data', prediction_dir=f'{args.directory}/predicted_demand/', result_dir=args.directory)
                if args.benchmark:
                    idq.get_rnn_inventory_decisions(station_id=station, date_list=date_list, hour_range=range(0,24), quarter=args.interval, model_type=['poisson_rnn', 'lr'], data_dir='data', prediction_dir=f'{args.directory}/predicted_demand/', result_dir=args.directory)
                    idq.get_benchmark_inventory_decisions(station, date_list, hour_range=range(0,24), quarter=args.interval, data_dir='data', result_dir=args.directory)
                print(f'Station {station} decision calculation finished!')
                # Evaluation  
                idq.get_inventory_decision_evaluation_results(station, date_list, hour_range=range(0,24), quarter=args.interval, model_type=['so_rnn',], flag_benchmark=args.benchmark, data_dir='data', result_dir=args.directory)
    except:
        pass
