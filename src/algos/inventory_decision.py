"""
Inventory Decision Model
-------
This file contains TODO. In particular, we implement:
(1) TODO
    Description
(2) TODO
    Description
(3) TODO
    Description
(4) TODO
    Description
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import math
import datetime
import scipy.stats as st

unit_cost_unsatisfied_pickup = 1
unit_cost_unsatisfied_return = 1

#generate 1.rnn forecast mean rate
def getForecastMeanRate(_pd, _date, _Hour, _Mean):
    _MeanRateList = []
    _date = pd.to_datetime(_date)
    _date_index = int(_pd[_pd['date'] == _date]['day_of_year'].mean())
    for i in _Hour:
        _hour_index = int(i + 24* (_date_index - 1))
        if _Mean[_hour_index-1][0] >= 0:
            _MeanRateList.extend(_Mean[_hour_index-1])
        else:
            _MeanRateList.extend([0.0,])
    return _MeanRateList

#generate 2.sample rate
def getSampleRate(_pd, _date, _Hour, _Mean, _Standard_deviation, _sample_time):
    _SampleRateList_all = []
    _date = pd.to_datetime(_date)
    _date_index = int(_pd[_pd['date'] == _date]['day_of_year'].mean())
    for t in range(_sample_time):
        _SampleRateList = []
        for i in _Hour:
            _hour_index = int(i + 24* (_date_index - 1))
            _SampleRateList.extend(np.random.normal(loc = _Mean[_hour_index-1], scale = _Standard_deviation[_hour_index-1]))
        _SampleRateList_all.append(_SampleRateList)
    return _SampleRateList_all

# generate 3.1 monthly historical average rate
def getMonthlyHistoricalMeanRate(_pd, _date, _Hour):
    _Pickup_HistoryRateList = []
    _Return_HistoryRateList = [] 
    _date = pd.to_datetime(_date)
    for i in _Hour:
        _historical_pickup = _pd[(_pd['date'] == _date) & (_pd['hour'] == i)]['monthly_historical_average_pickup'].values[0]
        _historical_return = _pd[(_pd['date'] == _date) & (_pd['hour'] == i)]['monthly_historical_average_return'].values[0]
        _Pickup_HistoryRateList.append(_historical_pickup)
        _Return_HistoryRateList.append(_historical_return)
    return _Pickup_HistoryRateList, _Return_HistoryRateList

# generate 3.2 yearly historical average rate
def getYearlyHistoricalMeanRate(_pd, _date, _Hour):
    _Pickup_HistoryRateList = []
    _Return_HistoryRateList = [] 
    _date = pd.to_datetime(_date)
    for i in _Hour:
        _historical_pickup = _pd[(_pd['date'] == _date) & (_pd['hour'] == i)]['historical_average_pickup'].values[0]
        _historical_return = _pd[(_pd['date'] == _date) & (_pd['hour'] == i)]['historical_average_return'].values[0]
        _Pickup_HistoryRateList.append(_historical_pickup)
        _Return_HistoryRateList.append(_historical_return)
    return _Pickup_HistoryRateList, _Return_HistoryRateList

# generate 4.true rate
def getTrueRate(_pd, _date, _Hour):
    _Pickup_TrueRateList = []
    _Return_TrueRateList = [] 
    _date = pd.to_datetime(_date)
    _selected_day = _pd[_pd['date'] == _date].reset_index(drop=True)
    for i in _Hour:
        _true_pickup = int(_selected_day[_selected_day['hour'] == i]['count_pickup'].values[0])
        _true_return = int(_selected_day[_selected_day['hour'] == i]['count_return'].values[0])
        _Pickup_TrueRateList.append(_true_pickup)
        _Return_TrueRateList.append(_true_return)
    return _Pickup_TrueRateList, _Return_TrueRateList

# Create transition rate matrix
def createTransitionRateMatrix(_pickupRate, _returnRate, _C):
    _Q = np.zeros((_C+1,_C+1))
    _Q[0,0] = - _returnRate
    _Q[0,1] = _returnRate
    _Q[_C,_C-1] = _pickupRate 
    _Q[_C,_C] = - _pickupRate
    for i in range(1, _C):
        for j in range(0,_C+1):
            if i == j:
                _Q[i,j] = - (_pickupRate + _returnRate)
            if j == i + 1:
                _Q[i,j] = _returnRate
            if j == i - 1:
                _Q[i,j] = _pickupRate
    return _Q

# Expected probability calculation
def rungeKutta(_Q, _pi0, _t = 1, _eta = 500):
    _h = _t/_eta
    _pi = _pi0
    _pi_avg = _pi
    for i in range(0, _eta):
        _k1 = _pi @ _Q
        _k2 = (_pi + _h * _k1/2) @ _Q
        _k3 = (_pi + _h * _k2/2) @ _Q
        _k4 = (_pi + _h * _k3) @ _Q
        _pi = _pi + _h*(_k1 + 2*_k2 + 2*_k3 + _k4)/6
        _pi_avg += _pi
    _pi_avg = (_pi_avg - _pi) / _eta
    return _pi, _pi_avg

def calExpectedCost(_pickupRates, _returnRates, _C, _starting_inventory, _T, _t = 1):
    _eta = math.floor(_T/_t)
    if _eta != len(_pickupRates) or _eta != len(_returnRates):
        print('input length error!')
        return
    _expected_unsatisfied_pickup = np.zeros((_eta))
    _expected_unsatisfied_return = np.zeros((_eta))
    _pi0 = np.zeros((_C+1))
    _pi0[_starting_inventory] = 1
    
    for i in range(0, _eta):
        _Q = createTransitionRateMatrix(_pickupRates[i], _returnRates[i], _C)
        _pi_last,_pi_avg = rungeKutta(_Q, _pi0, _t)
        _expected_unsatisfied_pickup[i] = _pickupRates[i] * _pi_avg[0]
        _expected_unsatisfied_return[i] = _returnRates[i] * _pi_avg[-1]
        _pi0 = _pi_last  
    _total_cost = np.sum(_expected_unsatisfied_pickup) * unit_cost_unsatisfied_pickup + np.sum(_expected_unsatisfied_return) * unit_cost_unsatisfied_return
    return _total_cost

def calInventoryDecisionOptExpectedCost(_C, _pickupRates, _returnRates, _firstHour, _lastHour):  
    _lengthT = (_lastHour - _firstHour) + 1
    _inventory_decision = 0
    _min_cost = calExpectedCost(_pickupRates, _returnRates, _C, _inventory_decision, _lengthT, _t = 1)
    for state in range(1,_C+1): 
        _cost = calExpectedCost(_pickupRates, _returnRates, _C , state, _lengthT, _t = 1)
        if _cost < _min_cost:
            _min_cost = _cost
            _inventory_decision = state
        if _cost > _min_cost + 0.000001:
            break
    return _inventory_decision

def calSampleDecisionOptExpectedCost(_sampleTime, _C, _firstHour, _lastHour, _pickup_rate_sample, _return_rate_sample):
    _lengthT = (_lastHour - _firstHour) + 1
    _cost_array = np.zeros((_sampleTime,_C+1), dtype= np.float)
    for i in range(_sampleTime):
        for inv in range(_C+1):
            _cost_array[i][inv] = calExpectedCost(_pickup_rate_sample[i], _return_rate_sample[i], _C, inv, _lengthT)
    _expected_cost = _cost_array.mean(axis = 0)
    _inventory_decision = np.argmin(_expected_cost)
    return _inventory_decision

def calRealCost(_C, _starting_inventory, _booking_data):
    _count_unsatisfied_pickup = 0
    _count_unsatisfied_return = 0
    _current_inventory = _starting_inventory
    for delta in _booking_data:
        if delta < 0:
            if _current_inventory == 0:
                _count_unsatisfied_pickup += 1
            else:
                _current_inventory += delta
        elif delta > 0:
            if _current_inventory == _C:
                _count_unsatisfied_return += 1
            else:
                _current_inventory += delta
    _total_cost = _count_unsatisfied_pickup * unit_cost_unsatisfied_pickup + _count_unsatisfied_return * unit_cost_unsatisfied_return
    return _total_cost

def calInventoryDecisionObjRealCost(_C, _booking_oneday, _firstHour, _lastHour):
    _booking_period = _booking_oneday[(_booking_oneday.hour >= _firstHour) & (_booking_oneday.hour <= _lastHour)].reset_index(drop=True)
    _booking_data  = np.array(_booking_period['inventory_change'])
    _cost_array = np.zeros(_C+1)
    for inv in range(_C+1):
        _cost_array[inv] = calRealCost(_C, inv, _booking_data)
    _inventory_decision = np.argmin(_cost_array)
    return _inventory_decision

def getDifferentDemandRates(_pd, _date, _Hour):
    Pickup_rate = {}
    Return_rate = {}
    Pickup_rate['specific_ha'], Return_rate['specific_ha'] = getMonthlyHistoricalMeanRate(_pd, _date, _Hour)
    Pickup_rate['overall_ha'], Return_rate['overall_ha'] = getYearlyHistoricalMeanRate(_pd, _date, _Hour)
    Pickup_rate['true_count'], Return_rate['true_count'] = getTrueRate(_pd, _date, _Hour)
    # if _sample_time > 0:
    #     Pickup_rate['sampling'] = getSampleRate(_pd, _date, _Hour, _Pickup_Forecast_Mean, _Pickup_Forecast_std_dev, _sample_time)
    #     Return_rate['sampling'] = getSampleRate(_pd, _date, _Hour, _Return_Forecast_Mean, _Return_Forecast_std_dev, _sample_time)
    return Pickup_rate, Return_rate

def getDifferentInventoryDecisionsObjCost(_C, _pickupRates, _returnRates, _booking_oneday, _firstHour, _lastHour, _sampleTime=0):
    inventory_decision = {}
    for key in _pickupRates.keys():
        if key == 'sampling':
            inventory_decision[key] = calSampleDecisionOptExpectedCost(_sampleTime, _C, _firstHour, _lastHour, _pickupRates[key], _returnRates[key])
        else:
            inventory_decision[key] = calInventoryDecisionOptExpectedCost(_C, _pickupRates[key], _returnRates[key], _firstHour, _lastHour)
    inventory_decision['best_possible'] = calInventoryDecisionObjRealCost(_C, _booking_oneday, _firstHour, _lastHour)
    return inventory_decision

# wrap function for all days
def getAllDaysExpectedCosts(_pickupRates, _returnRates, _inventory_decision, _C, _date_list, _time_frame):
    Costs_month = []
    _lengthT = (_time_frame[-1] - _time_frame[0]) + 1
    for dt in range(len(_date_list)):
        cost = calExpectedCost(_pickupRates[dt], _returnRates[dt], _C, _inventory_decision[dt], _lengthT, _t = 1)
        Costs_month.append(cost)
    return Costs_month

def getAllDaysRealCosts(_inventory_decision, _capacity, _booking_pd, _date_list, _time_frame):
    Costs_month = []
    for dt in range(len(_date_list)):
        booking_oneday = _booking_pd[_booking_pd.date == _date_list[dt]].reset_index(drop=True)
        booking_period = booking_oneday[(booking_oneday.hour >= _time_frame[0]) & (booking_oneday.hour <= _time_frame[-1])].reset_index(drop=True)
        booking_array = np.array(booking_period['inventory_change'])
        cost = calRealCost(_capacity, _inventory_decision[dt], booking_array)
        Costs_month.append(cost)
    return Costs_month

def get_inventory_decisions(station_id, date_list, hour_range, sample_time = 0, model_type='benchmarks', data_dir='data', result_dir='saved_files'):
    #read and preprocess data
    print("a")
    df_booking = pd.read_csv(data_dir + f'/raw/201708-201807/{station_id}_allbooking_2017.csv')
    df_booking['date'] = pd.to_datetime(df_booking['date']).dt.date
    print("a")
    df_demand = pd.read_csv(data_dir + f'/hourly_demand/201708-201807/{station_id}_hourlyRatesByDay_2017.csv')
    df_demand['date'] = pd.to_datetime(df_demand['date'])
    df_demand['day_of_year'] = df_demand['date'].dt.dayofyear
    df_demand['date'] = df_demand['date'].dt.date
    print("a")
    station_info = pd.read_csv(data_dir + '/raw/station_information_citibike.csv')
    capacity = int(station_info[station_info['id'] == int(station_id)]['capacity'])
    print("a")
    # get inventory decision over test dates:
    if model_type in ['mo_rnn', 'so_rnn', 'lr', 'so_rnn_2', "_test", "mo_rnn_fullcovar", "mo_rnn_diff"]:
        pickup_rate_mean = np.load(result_dir + f'/predicted_demand/{model_type}_{station_id}_pickup_mean.npy')
        dropoff_rate_mean = np.load(result_dir + f'/predicted_demand/{model_type}_{station_id}_return_mean.npy')
        if pickup_rate_mean.ndim == 1:
            pickup_rate_mean = np.load(result_dir + f'/predicted_demand/{model_type}_{station_id}_pickup_mean.npy')[:, None]
        if dropoff_rate_mean.ndim == 1:
            dropoff_rate_mean = np.load(result_dir + f'/predicted_demand/{model_type}_{station_id}_return_mean.npy')[:, None]
        print("a")
        # pickup_rate_std_dev = np.load(result_dir + f'/predicted_demand/{model_type}_{station_id}_pickup_std.npy')[:, None]
        # dropoff_rate_std_dev = np.load(result_dir + f'/predicted_demand/{model_type}_{station_id}_return_std.npy')[:, None]
        inventory_decision_dict_list = []
        for dt in date_list:
            print("start calculating for date ", str(dt))
            pickup_rate_dict = {}
            return_rate_dict = {}
            pickup_rate_dict[model_type] = getForecastMeanRate(df_demand, dt, hour_range, pickup_rate_mean)
            return_rate_dict[model_type] = getForecastMeanRate(df_demand, dt, hour_range, dropoff_rate_mean)     
            inventory_decision_oneday_dict = {}
            for key in pickup_rate_dict.keys():
               inventory_decision_oneday_dict[key] = calInventoryDecisionOptExpectedCost(capacity, pickup_rate_dict[key], return_rate_dict[key], hour_range[0], hour_range[-1])
            inventory_decision_dict_list.append(inventory_decision_oneday_dict)
        print(f"\n {model_type} inventory decision calculation finished!")
    elif model_type == 'benchmarks':
        inventory_decision_dict_list = []
        for dt in date_list:
            print("start calculating for date ", str(dt))
            pickup_rate_dict, return_rate_dict = getDifferentDemandRates(df_demand, dt, hour_range)
            df_booking_oneday = df_booking[df_booking.date == dt].reset_index(drop=True)
            inventory_decision_oneday_dict = \
            getDifferentInventoryDecisionsObjCost(capacity, pickup_rate_dict, return_rate_dict, 
            df_booking_oneday, hour_range[0], hour_range[-1], sample_time)
            inventory_decision_dict_list.append(inventory_decision_oneday_dict)
        print(f"\n {model_type} inventory decision calculation finished!")
    else:
        raise TypeError('Un-recognized model type, check your spelling!') 
    # save decisions:
    inventory_decision_dict = {}
    for key in inventory_decision_dict_list[0].keys():
        inventory_decision = np.zeros(len(date_list))
        for i in range(len(date_list)):
            inventory_decision[i] = inventory_decision_dict_list[i][key]
        inventory_decision_dict[key] = inventory_decision
        np.save(result_dir + f'/inventory_decisions/rnn/{station_id}_inventory_decision_' + key + '.npy', inventory_decision_dict[key])
#     print("\nDecisions are saved in saved_files/inventory_decisions/rnn/")
    return inventory_decision_dict
        
def get_inventory_decisions_folds(station_id, date_list, hour_range, sample_time = 0, model_type='benchmarks', data_dir='data', result_dir='saved_files', folds=False):
    #read and preprocess data
    folds = 4 if folds else 1
    df_booking = pd.read_csv(data_dir + f'/raw/201708-201807/{station_id}_allbooking_2017.csv')
    df_booking['date'] = pd.to_datetime(df_booking['date']).dt.date
    
    df_demand = pd.read_csv(data_dir + f'/hourly_demand/201708-201807/{station_id}_hourlyRatesByDay_2017.csv')
    df_demand['date'] = pd.to_datetime(df_demand['date'])
    df_demand['day_of_year'] = df_demand['date'].dt.dayofyear
    df_demand['date'] = df_demand['date'].dt.date
    
    station_info = pd.read_csv(data_dir + '/raw/station_information_citibike.csv')
    capacity = int(station_info[station_info['id'] == int(station_id)]['capacity'])
    # get inventory decision over test dates:
    for fold in range(folds):
        if model_type in ['mo_rnn', 'so_rnn', 'lr', 'so_rnn_2', "_test", "mo_rnn_fullcovar", "mo_rnn_diff"]:
            pickup_rate_mean = np.load(result_dir + f'/predicted_demand/{model_type}_{station_id}_pickup_fold{fold}_mean.npy')
            dropoff_rate_mean = np.load(result_dir + f'/predicted_demand/{model_type}_{station_id}_return_fold{fold}_mean.npy')
            if pickup_rate_mean.ndim == 1:
                pickup_rate_mean = np.load(result_dir + f'/predicted_demand/{model_type}_{station_id}_pickup_fold{fold}_mean.npy')[:, None]
            if dropoff_rate_mean.ndim == 1:
                dropoff_rate_mean = np.load(result_dir + f'/predicted_demand/{model_type}_{station_id}_return_fold{fold}_mean.npy')[:, None]
            # pickup_rate_std_dev = np.load(result_dir + f'/predicted_demand/{model_type}_{station_id}_pickup_std.npy')[:, None]
            # dropoff_rate_std_dev = np.load(result_dir + f'/predicted_demand/{model_type}_{station_id}_return_std.npy')[:, None]
            inventory_decision_dict_list = []
            for dt in date_list:
                print("start calculating for date ", str(dt))
                pickup_rate_dict = {}
                return_rate_dict = {}
                pickup_rate_dict[model_type] = getForecastMeanRate(df_demand, dt, hour_range, pickup_rate_mean)
                return_rate_dict[model_type] = getForecastMeanRate(df_demand, dt, hour_range, dropoff_rate_mean)     
                inventory_decision_oneday_dict = {}
                for key in pickup_rate_dict.keys():
                   inventory_decision_oneday_dict[key] = calInventoryDecisionOptExpectedCost(capacity, pickup_rate_dict[key], return_rate_dict[key], hour_range[0], hour_range[-1])
                inventory_decision_dict_list.append(inventory_decision_oneday_dict)
            print(f"\n {model_type} inventory decision calculation finished!")
        elif model_type == 'benchmarks':
            inventory_decision_dict_list = []
            for dt in date_list:
                print("start calculating for date ", str(dt))
                pickup_rate_dict, return_rate_dict = getDifferentDemandRates(df_demand, dt, hour_range)
                df_booking_oneday = df_booking[df_booking.date == dt].reset_index(drop=True)
                inventory_decision_oneday_dict = \
                getDifferentInventoryDecisionsObjCost(capacity, pickup_rate_dict, return_rate_dict, 
                df_booking_oneday, hour_range[0], hour_range[-1], sample_time)
                inventory_decision_dict_list.append(inventory_decision_oneday_dict)
            print(f"\n {model_type} inventory decision calculation finished!")
        else:
            raise TypeError('Un-recognized model type, check your spelling!') 
        # save decisions:
        inventory_decision_dict = {}
        for key in inventory_decision_dict_list[0].keys():
            inventory_decision = np.zeros(len(date_list))
            for i in range(len(date_list)):
                inventory_decision[i] = inventory_decision_dict_list[i][key]
            inventory_decision_dict[key] = inventory_decision
            np.save(result_dir + f'/inventory_decisions/rnn/{station_id}_fold{fold}_inventory_decision_' + key + '.npy', inventory_decision_dict[key])
    #     print("\nDecisions are saved in saved_files/inventory_decisions/rnn/")
        return inventory_decision_dict

        
