import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import os
from pyro.infer import Trace_ELBO
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Trace_ELBO_Wrapper(Trace_ELBO):
    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the (Negative) ELBO, KL divergence and Marginal Log-Likelihood.
        :rtype: float
        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        """
        elbo = 0.0
        log_prob_sum = 0.0
        kl_sum = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            log_prob = model_trace.log_prob_sum()
            log_prob_sum += log_prob
            kl = guide_trace.log_prob_sum()
            kl_sum += kl
            elbo_particle = log_prob - kl
            elbo += elbo_particle / self.num_particles
        loss = -elbo
        return loss, kl_sum, log_prob_sum
    
    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the (Negative) ELBO, KL divergence and Marginal Log-Likelihood.
        :rtype: float
        Computes the ELBO as well as the surrogate ELBO that is used to form the gradient estimator.
        Performs backward on the latter. Num_particle many samples are used to form the estimators.
        """
        loss = 0.0
        log_prob_sum = 0.0
        kl_sum = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            log_prob = model_trace.log_prob_sum()
            log_prob_sum += log_prob / self.num_particles
            kl = guide_trace.log_prob_sum()
            kl_sum += kl / self.num_particles
            loss_particle, surrogate_loss_particle = self._differentiable_loss_particle(model_trace, guide_trace)
            loss += loss_particle / self.num_particles
            
            # collect parameters to train from model and guide
            trainable_params = any(site["type"] == "param"
                                   for trace in (model_trace, guide_trace)
                                   for site in trace.nodes.values())

            if trainable_params and getattr(surrogate_loss_particle, 'requires_grad', False):
                surrogate_loss_particle = surrogate_loss_particle / self.num_particles
                surrogate_loss_particle.backward(retain_graph=self.retain_graph)
                
        warn_if_nan(loss, "loss")
        return loss, kl_sum, log_prob_sum
    
def get_performance_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

def read_and_preprocess_data(demand_path="data/rate_count/229_hourlyRatesByDay_2018.csv", weather_path="data/weather/weather2018_NYC_Shared.csv", station_229=True, 
                            interval=60):
    # read raw data
    df_demand = pd.read_csv(demand_path, parse_dates=['date'])
    df_weather = pd.read_csv(weather_path)
    # create weekday feature
    df_demand['weekday'] = df_demand.date.dt.dayofweek.values
    # separate pickup & dropoffs
    df_pickup = df_demand[["count_pickup", "hour", "weekday"]]
    df_dropoff = df_demand["count_return"]
    # get one-hot-encoded DayOfWeek and TimeOfDay features
    df_pickup = pd.get_dummies(df_pickup, columns=["weekday"])
    df_pickup = pd.get_dummies(df_pickup, columns=["hour"])
    assert interval in [15, 30, 60]
    multiplier = int(60 / interval)
    df_station = np.concatenate((df_dropoff.values.reshape(-1,1),
                                 df_pickup.values[:, 0].reshape(-1,1),
                                 df_dropoff.values.reshape(-1,1) - df_pickup.values[:, 0].reshape(-1,1),
                                 df_pickup.values[:,1:].reshape(-1,df_pickup.shape[1]-1),
                                 df_weather.precip_prob.values.reshape(-1,1)), axis=1)
    # create torch.tensors
    y = df_station[:,:3]
    X_tensor = torch.from_numpy(df_station[:,3:]).float()
    y_tensor = torch.from_numpy(y).float()
    
    X_tensor = X_tensor[~torch.any(torch.isnan(y_tensor),dim=1)]
    y_tensor = y_tensor[~torch.any(torch.isnan(y_tensor),dim=1)]
    # Define train/valid/test split
    # 70% for training, Nov + Dec (i.e. 61 days * 24 hours/days) for testing, remaining for validation
    train_perc = 0.70
    train_idx = np.arange(int(len(y_tensor)*train_perc))
    valid_idx = np.arange(int(len(y_tensor)*train_perc), int(len(y_tensor)-(61*24*multiplier)))
    test_idx = np.arange(int(len(y_tensor)-(61*24*multiplier)), len(y_tensor))

    X_train, X_valid, X_test = X_tensor[train_idx], X_tensor[valid_idx], X_tensor[test_idx]
    y_train, y_valid, y_test = y_tensor[train_idx], y_tensor[valid_idx], y_tensor[test_idx]
    return df_station, X_train, X_valid, X_test, y_train, y_valid, y_test, X_tensor, y_tensor

def get_results(model, X, y, results_path="saved_files", model_type="mo_rnn", labels=["return", "pickup"], write_mode="w", station=None, interval=60):
    assert interval in [15, 30, 60]
    multiplier = int(60/interval)
    train_perc = 0.7
    train_idx = np.arange(int(len(y)*train_perc))
    valid_idx = np.arange(int(len(y)*train_perc), int(len(y)-(61*24*multiplier)))
    test_idx = np.arange(int(len(y)-(61*24*multiplier)), len(y))
    y_train, y_valid, y_test = y[train_idx].numpy(), y[valid_idx].numpy(), y[test_idx].numpy()
    with open(results_path + f"/results/prediction/" + f"output_st{station}_{model_type}.txt", write_mode) as text_file:
        for i, label in enumerate(labels):
            if model_type in ['so_rnn', 'mo_rnn', 'mo_rnn_fullcovar', 'mo_rnn_diff']: 
                log_lambda_loc, log_lambda_scale = model.guide(X, y, forecast=True)
                if model_type == 'mo_rnn_fullcovar':
                    log_lambda_scale = torch.diagonal(log_lambda_scale[:, 0], dim1=1, dim2=2)
                log_lambda_loc = log_lambda_loc[:, i]
                log_lambda_scale = log_lambda_scale[:, i]
                lambda_loc = torch.exp(log_lambda_loc + log_lambda_scale**2 / 2).detach()
                lambda_scale = torch.sqrt((torch.exp(log_lambda_scale**2) - 1) * (torch.exp(2*log_lambda_loc + log_lambda_scale**2))).detach()
                np.save(results_path + "/predicted_demand/" + f"{model_type}_{station}_{label}_mean.npy", lambda_loc.detach().numpy())
                np.save(results_path + "/predicted_demand/" + f"{model_type}_{station}_{label}_std.npy", lambda_scale.detach().numpy())
                rmse, mae, r2 = get_performance_metrics(y_train[1:, i], lambda_loc[:len(y_train)-1])
                print(f" STATION {station} - {label}", file=text_file)
                print("---------------\n", file=text_file)
                print(f"| TRAIN |  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R^2: {r2:.2f}\n", file=text_file)
                rmse, mae, r2 = get_performance_metrics(y_valid[:, i], lambda_loc[len(y_train)-1:len(y_train) + len(y_valid)-1])
                print(f"| VALIDATION |  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R^2: {r2:.2f}\n", file=text_file)
                rmse, mae, r2 = get_performance_metrics(y_test[:, i], lambda_loc[test_idx-1])
                print(f"| TEST |  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R^2: {r2:.2f}\n", file=text_file)
            elif model_type=='lr': 
                y_pred = model(X).detach().numpy()
                y_pred[y_pred <= 0.] = torch.zeros((1,))
                np.save(results_path + "/predicted_demand/" + f"{model_type}_{station}_{label}_mean.npy", y_pred)
                rmse, mae, r2 = get_performance_metrics(y_train[1:, i], y_pred[:len(y_train)-1])
                print(f" STATION {station} - {label}", file=text_file)
                print("---------------\n", file=text_file)
                print(f"| TRAIN |  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R^2: {r2:.2f}\n", file=text_file)
                rmse, mae, r2 = get_performance_metrics(y_valid[:, i], y_pred[len(y_train)-1:len(y_train) + len(y_valid)-1])
                print(f"| VALIDATION |  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R^2: {r2:.2f}\n", file=text_file)
                rmse, mae, r2 = get_performance_metrics(y_test[:, i], y_pred[test_idx-1])
                print(f"| TEST |  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R^2: {r2:.2f}\n", file=text_file)
                lambda_loc = y_pred.flatten()
            elif model_type=='poisson_rnn':
                log_lambda_loc = model.model(X, y, forecast=True).detach().numpy()
                lambda_loc = log_lambda_loc[:, i]
                np.save(results_path + "/predicted_demand/" + f"{model_type}_{station}_{label}_mean.npy", lambda_loc)
                rmse, mae, r2 = get_performance_metrics(y_train[1:, i], lambda_loc[:len(y_train)-1])
                print(f" STATION {station} - {label}", file=text_file)
                print("---------------\n", file=text_file)
                print(f"| TRAIN |  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R^2: {r2:.2f}\n", file=text_file)
                rmse, mae, r2 = get_performance_metrics(y_valid[:, i], lambda_loc[len(y_train)-1:len(y_train) + len(y_valid)-1])
                print(f"| VALIDATION |  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R^2: {r2:.2f}\n", file=text_file)
                rmse, mae, r2 = get_performance_metrics(y_test[:, i], lambda_loc[test_idx-1])
                print(f"| TEST |  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R^2: {r2:.2f}\n", file=text_file)
                lambda_loc = lambda_loc.flatten()
            else:
                raise TypeError('Un-recognized model type, check your spelling!') 
                
        # Prediction Viz.
            n = 500
            fig = plt.figure(figsize=(30,6))
            plt.plot(lambda_loc[test_idx[:n]-1], label="prediction")
            plt.plot(y_test[:n,i], label="data")
            if model_type not in ["lr", "poisson_rnn"]:
                plt.fill_between(np.arange(len(lambda_loc[test_idx[:n]])), (lambda_loc[test_idx[:n]-1] - 1.96*lambda_scale[test_idx[:n]-1]).flatten(),
                                 (lambda_loc[test_idx[:n]-1] + 1.96*lambda_scale[test_idx[:n]-1]), alpha=0.3)
            plt.xlabel("hour", fontsize=25)
            plt.ylabel(f"{labels[i]} count", fontsize=25)
            plt.tick_params(axis='both', labelsize=25)
            plt.legend(fontsize=25);
            plt.title(f"{labels[i]} predictions", fontsize=25)
            fig.savefig(results_path + "/images/" + f"{labels[i]}_{station}_{model_type}_preds.png", dpi=fig.dpi)

            # Residual Viz.
            fig = plt.figure(figsize=(10,12))
            ax2 = plt.subplot(211)
            sns.distplot(lambda_loc[test_idx-1] - y_test[:, i])
            plt.title(f"{labels[i]} residual distribution")
            plt.xlabel(r"$y_i - \hat{y}_i$")
            plt.ylabel("prob")
            ax4 = plt.subplot(212)
            plt.plot(lambda_loc[test_idx-1] - y_test[:, i])
            plt.hlines((lambda_loc[test_idx-1] - y_test[:, i]).max(), 0, len(y_test), linestyles="--", color="k")
            plt.hlines((lambda_loc[test_idx-1] - y_test[:, i]).min(), 0, len(y_test), linestyles="--", color="k")
            plt.title(f"{labels[i]} residual plot")
            plt.ylabel(r"$y_i - \hat{y}_i$")
            plt.xlabel("t")
            fig.savefig(results_path + "/images/" + f"{labels[i]}_{station}_{model_type}_residuals.png", dpi=fig.dpi)

