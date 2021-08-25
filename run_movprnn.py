"""
Executing MOVP-RNN
------
This file orchestrates various use cases for the Multi-Output Variational Poisson-RNN (MOVP-RNN) as introduced in Section 4.1 of the paper.

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
from src.algos.vprnn import MOVPRNN
from src.algos.inventory_decision import get_inventory_decisions
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
parser.add_argument('--sample-time', type=int, default=0, metavar='N',
                    help='number of posterior predictive samples (default: 0)')
parser.add_argument('--no-decision', type=bool, default=False,
                    help='disables decision model')

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
    args.stations = [151, 168, 285, 293, 3263, 327, 3435, 358, 359, 3641,
                     368, 3711, 387, 402, 426, 435, 445, 446, 453, 462, 482, 491,
                     497, 499, 504, 514, 519, 229
                    ]
# loop over selected stations (i.e., execute full pipeline) 
for station in args.stations:
    try:
        df_station, X_train, X_valid, X_test, y_train, y_valid, y_test, X_tensor, y_tensor = \
                 read_and_preprocess_data(demand_path=f"data/demand_rate/{str(args.interval)}min/{str(station)}_{args.file_interval}RatesByDay_2018.csv",
                                        weather_path=f"data/raw/weather2018_{args.interval}min.csv", station_229=False, interval=args.interval)
        y_train, y_valid, y_test, y_tensor = y_train[:, :2], y_valid[:, :2], y_test[:, :2], y_tensor[:, :2]
        
        if args.train:
            print(f"\n Training started for St. {station}, with patience={args.patience}")
            # train process
            movprnn = MOVPRNN(input_dim=32, output_dim=2, p_model_dim=128, p_model_layers=1, 
                        q_model_dim=128, q_model_layers=1).to(device)
            svi = SVI(movprnn.model, movprnn.guide, pyro.optim.RMSprop({"lr": 0.001}), Trace_ELBO(num_particles=1))

            train_losses = []
            valid_losses = []
            pyro.clear_param_store()
            epochs = tqdm.trange(args.epochs)
            best_loss = np.inf
            patience = args.patience
            patience_counter = 0

            for epoch in epochs:
                movprnn.train()
                loss = 0
                b = 0
                X_train, y_train = X_train.to(device), y_train.to(device)
                X_valid, y_valid = X_valid.to(device), y_valid.to(device)
                torch.nn.utils.clip_grad_norm_(movprnn.parameters(),5.)
                loss += svi.step(X_train, y_train) / len(X_train)
                movprnn.eval()
                valid_loss = svi.evaluate_loss(X_valid, y_valid) / len(X_valid)
                epochs.set_description("Train ELBO: {:.3f} - Test ELBO: {:.3f} - Patience: {:.0f}".format(loss, valid_loss, patience_counter))
                if valid_loss <= best_loss:
                    torch.save(movprnn.state_dict(), f"{args.directory}/trained_models/rnn/mo_st{station}")
                    best_loss = valid_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"\n Training finished for St. {station}, loss converged!")
                        break
                train_losses.append(float(loss))
                valid_losses.append(float(valid_loss))
                b += 1
                del loss
                del valid_loss
                # TODO: plot, save training losses?
            if args.predict:
            # create model instance and load optimal pre-trained parameters
                movprnn = MOVPRNN(input_dim=32, output_dim=2, p_model_dim=128, p_model_layers=1, 
                        q_model_dim=128, q_model_layers=1)
                movprnn.load_state_dict(torch.load(f"{args.directory}/trained_models/rnn/mo_st{station}", map_location=device))	

                get_results(model=movprnn, X=X_tensor, y=y_tensor, station=station, results_path=args.directory, interval=args.interval,
                        model_type="mo_rnn")

############################################################
############################################################
########## TODO: Fix decision pipeline from here ###########
############################################################
############################################################

        # define test dates
        cal = NewYork()
        start_date = datetime.date(2018,11,1)
        end_date = datetime.date(2018,12,31)
        date_list = []
        for n in range((end_date - start_date).days + 1):
            dt = start_date + datetime.timedelta(days=n)
#             if dt.isoweekday() not in (6,7) and cal.is_working_day(dt):
            date_list.append(dt)
        if args.decision:
            inventory_decision_dict = \
                get_inventory_decisions(station_id=station, date_list=date_list, 
                hour_range=range(0,24), model_type='mo_rnn', result_dir=args.directory)
    except KeyboardInterrupt:
        pass