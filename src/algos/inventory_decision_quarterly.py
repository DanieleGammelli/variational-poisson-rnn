"""
Inventory Decision Model with 15-, 30-minute intervals
-------
This file contains the inventory decision model specifications. In particular, we implement:
(1) Inputs: 
    Reads demand predictions as inputs to the inventory decision model (Section 3.1)
(2) Inventory decision calculation:
    Step 1. Calculates the transient probability p(s, \delta, t) (Eq.1, Section 3.1)
    Step 2. Calculates the UDF and selects the optimal starting inventory s∗ which minimizes the UDF (Eq.2, Section 3.1)
    Step 3. (Optional): Calulates benchmark decisions
(3) Evaluation:
    Calculates prescriptive results (Section 4.2)
    
"""

import pandas as pd
import numpy as np
import math

# (1) Input demand predictions
#generate 1. rnn forecast mean rate
def getForecastMeanRate(_date, _hour, _quarter, _forecast_mean):
    mean_rate_list = []
    date_index = pd.to_datetime(_date).dayofyear
    start_time_index = int(_hour[0]* len(_quarter) + 24* len(_quarter)*(date_index - 1))
    end_time_index = int(start_time_index + len(_hour) * len(_quarter))
    
    for i in range(start_time_index, end_time_index):    
        if _forecast_mean[i-1] >= 0:
            mean_rate_list.append(_forecast_mean[i-1])
        else:
            mean_rate_list.append(0)
    return mean_rate_list

# generate 2. moving average rate
def getMonthlyHistoricalMeanRate(_df, _date, _hour, _quarter):
    pickup_rate_list = []
    return_rate_list = [] 
    _date = pd.to_datetime(_date)
    for i in _hour:
        for j in _quarter:
            historical_pickup = _df[(_df['date'] == _date) & (_df['hour'] == i) & (_df['hourQuarter'] == j)]['monthly_historical_average_pickup'].values[0]
            historical_return = _df[(_df['date'] == _date) & (_df['hour'] == i) & (_df['hourQuarter'] == j)]['monthly_historical_average_return'].values[0]
            pickup_rate_list.append(historical_pickup)
            return_rate_list.append(historical_return)
    return pickup_rate_list, return_rate_list

# generate 3. historical average rate
def getYearlyHistoricalMeanRate(_df, _date, _hour, _quarter):
    pickup_rate_list = []
    return_rate_list = [] 
    _date = pd.to_datetime(_date)
    for i in _hour:
        for j in _quarter:
            historical_pickup = _df[(_df['date'] == _date) & (_df['hour'] == i) & (_df['hourQuarter'] == j)]['historical_average_pickup'].values[0]
            historical_return = _df[(_df['date'] == _date) & (_df['hour'] == i) & (_df['hourQuarter'] == j)]['historical_average_return'].values[0]
            pickup_rate_list.append(historical_pickup)
            return_rate_list.append(historical_return)
    return pickup_rate_list, return_rate_list

# generate 4. true count rate
def getTrueRate(_df, _date, _hour, _quarter):
    pickup_rate_list = []
    return_rate_list = [] 
    _date = pd.to_datetime(_date)
    _selected_day = _df[_df['date'] == _date].reset_index(drop=True)
    for i in _hour:
        for j in _quarter:
            true_pickup = int(_selected_day[(_selected_day['hour'] == i) & (_selected_day['hourQuarter'] == j)]['count_pickup'].values[0])
            true_return = int(_selected_day[(_selected_day['hour'] == i) & (_selected_day['hourQuarter'] == j)]['count_return'].values[0])
            pickup_rate_list.append(true_pickup)
            return_rate_list.append(true_return)
    return pickup_rate_list, return_rate_list

# (2) Inventory decision calculation:
# Create transition rate matrix
def createTransitionRateMatrix(_pickupRate, _returnRate, _C):
    _Q = np.zeros((_C+1,_C+1))
    _Q[0,0] = -_returnRate
    _Q[0,1] = _returnRate
    _Q[_C,_C-1] = _pickupRate 
    _Q[_C,_C] = -_pickupRate
    for i in range(1, _C):
        for j in range(0,_C+1):
            if i == j:
                _Q[i,j] = - (_pickupRate + _returnRate)
            if j == i + 1:
                _Q[i,j] = _returnRate
            if j == i - 1:
                _Q[i,j] = _pickupRate
    return _Q

# Step 1. Calculates the transient probability
def rungeKutta(_Q, _pi0, _quarter, _t = 1, _step = 500):
    _step = int(_step / len(_quarter))
    _h = _t/_step
    _pi = _pi0
    _pi_avg = _pi
    for j in range(0, _step):
        _k1 = _pi @ _Q
        _k2 = (_pi + _h * _k1/2) @ _Q
        _k3 = (_pi + _h * _k2/2) @ _Q
        _k4 = (_pi + _h * _k3) @ _Q
        _pi = _pi + _h*(_k1 + 2*_k2 + 2*_k3 + _k4)/6
        _pi_avg += _pi
    _pi_avg = (_pi_avg - _pi) / _step
    return _pi, _pi_avg

# Step 2. Calculates the UDF of each starting_inventory
# _t is stick to the length of time interval
def calExpectedCost(_pickupRates, _returnRates, _C, _starting_inventory, _T, _t, _quarter, unit_cost_unsatisfied_pickup=1, unit_cost_unsatisfied_return=1):
    _eta = math.ceil(_T/_t)
    if _eta != len(_pickupRates) or _eta != len(_returnRates):
        print('input length error!')
        return
    _expected_unsatisfied_pickup = np.zeros((_eta))
    _expected_unsatisfied_return = np.zeros((_eta))
    _pi0 = np.zeros((_C+1))
    _pi0[_starting_inventory] = 1
    
    for i in range(0, _eta):
        _Q = createTransitionRateMatrix(_pickupRates[i], _returnRates[i], _C)
        _pi_last,_pi_avg = rungeKutta(_Q, _pi0, _quarter, _t)
        _expected_unsatisfied_pickup[i] = _pickupRates[i] * _pi_avg[0]
        _expected_unsatisfied_return[i] = _returnRates[i] * _pi_avg[-1]
        _pi0 = _pi_last
    _total_cost = np.sum(_expected_unsatisfied_pickup) * unit_cost_unsatisfied_pickup + np.sum(_expected_unsatisfied_return) * unit_cost_unsatisfied_return
    return _total_cost

# Select the optimal starting inventory s∗ which minimizes the UDF
def calInventoryDecisionOptExpectedCost(_C, _pickupRates, _returnRates, _lengthT, _t, _quarter):  
    _inventory_decision = 0
    _min_cost = calExpectedCost(_pickupRates, _returnRates, _C, _inventory_decision, _lengthT, _t, _quarter)
    for state in range(1,_C+1): 
        _cost = calExpectedCost(_pickupRates, _returnRates, _C , state, _lengthT, _t, _quarter)
        if _cost < _min_cost:
            _min_cost = _cost
            _inventory_decision = state
        elif _cost > _min_cost + 0.000001:
            break
    return _inventory_decision, _min_cost

def readStationData(station_id, _Quarter, data_dir = '../data'):
    #Read capacity
    station_info = pd.read_csv(data_dir + '/raw/station_information_citibike.csv')
    capacity = int(station_info[station_info['id'] == int(station_id)]['capacity'])
    print('station ID:', station_id ,'\ncapacity:',capacity)
    #Read real count data
    df_booking = pd.read_csv(data_dir + f'/raw/{station_id}_allbooking_2018.csv')
    df_booking['date'] = pd.to_datetime(df_booking['date']).dt.date
    #Read quarter demand data
    if len(_Quarter) == 4:
        df_demand = pd.read_csv(data_dir + f'/demand_rate/15min/{station_id}_15minRatesByDay_2018.csv')
    elif len(_Quarter) == 2:
        df_demand = pd.read_csv(data_dir + f'/demand_rate/30min/{station_id}_30minRatesByDay_2018.csv')
    else:
        print("Error: Input Wrong Hour Quarter!")
        return
    df_demand['date'] = pd.to_datetime(df_demand['date'])
    df_demand['day_of_year'] = df_demand['date'].dt.dayofyear
    df_demand['date'] = df_demand['date'].dt.date
    return capacity, df_booking, df_demand

# Wrapper function
def get_rnn_inventory_decisions(station_id, date_list, hour_range, quarter, model_type, data_dir, prediction_dir, result_dir='saved_files'):
    quarter=range(int(60/quarter))
    capacity, df_booking, df_demand = readStationData(station_id, quarter, data_dir)
    #define the number of aggregations
    _lengthT = len(hour_range)*len(quarter)

    inventory_decision_dict = {key:[] for key in model_type}
    # min_expected_cost_dict = {key:[] for key in model_type}
    # calculate inventory decision over test dates:
    for key in model_type:
        pickup_rate_mean = np.load(prediction_dir + f'{key}_{station_id}_pickup_mean.npy')
        dropoff_rate_mean = np.load(prediction_dir + f'{key}_{station_id}_return_mean.npy')
        for dt in date_list:
            print("start calculating for date ", str(dt))
            pickup_rate_oneday = getForecastMeanRate(dt, hour_range, quarter, pickup_rate_mean)
            return_rate_oneday = getForecastMeanRate(dt, hour_range, quarter, dropoff_rate_mean)     
            inventory_decision_oneday, min_expected_cost_oneday = calInventoryDecisionOptExpectedCost(capacity, pickup_rate_oneday, return_rate_oneday, _lengthT, _t=1, _quarter = quarter)
            inventory_decision_dict[key].append(inventory_decision_oneday)
            # min_expected_cost_dict[key].append(min_expected_cost_oneday)
        np.save(result_dir + f'/inventory_decisions/{station_id}_inventory_decision_' + key + '.npy', inventory_decision_dict[key])
        # np.save(result_dir + f'/results/optimization/respective_expected_costs/{station_id}_expected_cost_' + key + '.npy', min_expected_cost_dict[key])
        print(f"\n {key} inventory decision calculation finished!")
    return inventory_decision_dict

# Step 3. (Optional): Calulates benchmark decisions
def calRealCost(_C, _starting_inventory, _booking_data, unit_cost_unsatisfied_pickup=1, unit_cost_unsatisfied_return=1):
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

# Oracle decision
def calInventoryDecisionObjRealCost(_C, _booking_oneday, _firstHour, _lastHour):
    _booking_period = _booking_oneday[(_booking_oneday.hour >= _firstHour) & (_booking_oneday.hour <= _lastHour)].reset_index(drop=True)
    _booking_data  = np.array(_booking_period['inventory_change'])
    _cost_array = np.zeros(_C+1)
    for inv in range(_C+1):
        _cost_array[inv] = calRealCost(_C, inv, _booking_data)
    _inventory_decision = np.argmin(_cost_array)
    return _inventory_decision 

# Other benchmark decisions
def getDifferentDemandRates(_df, _date, _hour, _quarter):
    Pickup_rate = {}
    Return_rate = {}
    Pickup_rate['specific_ha'], Return_rate['specific_ha'] = getMonthlyHistoricalMeanRate(_df, _date, _hour, _quarter)
    Pickup_rate['overall_ha'], Return_rate['overall_ha'] = getYearlyHistoricalMeanRate(_df, _date, _hour, _quarter)
    Pickup_rate['true_count'], Return_rate['true_count'] = getTrueRate(_df, _date, _hour, _quarter)
    # if _sample_time > 0:
    #     Pickup_rate['sampling'] = getSampleRate(_pd, _date, _Hour, _Pickup_Forecast_Mean, _Pickup_Forecast_std_dev, _sample_time)
    #     Return_rate['sampling'] = getSampleRate(_pd, _date, _Hour, _Return_Forecast_Mean, _Return_Forecast_std_dev, _sample_time)
    return Pickup_rate, Return_rate

def getDifferentInventoryDecisionsObjCost(_C, _pickupRates, _returnRates, _lengthT, _t, _quarter):
    inventory_decision = {}
    min_expected_cost = {}
    for key in _pickupRates.keys():
        inventory_decision[key], min_expected_cost[key] = calInventoryDecisionOptExpectedCost(_C, _pickupRates[key], _returnRates[key], _lengthT, _t, _quarter)
    return inventory_decision, min_expected_cost

# Wrapper function
def get_benchmark_inventory_decisions(station_id, date_list, hour_range, quarter, data_dir='data', result_dir='saved_files'):
    quarter=range(int(60/quarter))
    #read station data
    capacity, df_booking, df_demand = readStationData(station_id, quarter, data_dir)
    #define the number of aggregations
    _lengthT = len(hour_range)*len(quarter)

    inventory_decision_dict_list = []
    # min_expected_cost_dict_list = []
    for dt in date_list:
        print("start calculating for date ", str(dt))
        pickup_rate_dict, return_rate_dict = getDifferentDemandRates(df_demand, dt, hour_range, quarter)
        df_booking_oneday = df_booking[df_booking.date == dt].reset_index(drop=True)
        inventory_decision_oneday_dict, min_expected_cost_oneday_dict = \
        getDifferentInventoryDecisionsObjCost(capacity, pickup_rate_dict, return_rate_dict, _lengthT, _t = 1, _quarter=quarter)
        inventory_decision_oneday_dict['best_possible'] = calInventoryDecisionObjRealCost(capacity, df_booking_oneday, hour_range[0], hour_range[-1])
        inventory_decision_dict_list.append(inventory_decision_oneday_dict)
        # min_expected_cost_dict_list.append(min_expected_cost_oneday_dict)
    print(f"\n inventory decision calculation finished!")
    # save decisions:
    inventory_decision_dict = {}
    for key in inventory_decision_dict_list[0].keys():
        inventory_decision = np.zeros(len(date_list))
        min_expected_cost = np.zeros(len(date_list))
        for i in range(len(date_list)):
            inventory_decision[i] = inventory_decision_dict_list[i][key]
            # min_expected_cost[i] = min_expected_cost_dict_list[i][key]
        inventory_decision_dict[key] = inventory_decision
        np.save(result_dir + f'/inventory_decisions/{station_id}_inventory_decision_{key}.npy', inventory_decision_dict[key])
        # np.save(result_dir + f'/results/optimization/respective_expected_costs/{station_id}_expected_cost_{key}.npy', min_expected_cost)
    # save best possible decision:
    best_possible_decision = np.zeros(len(date_list))
    for i in range(len(date_list)):
        best_possible_decision[i] = inventory_decision_dict_list[i]['best_possible']
    np.save(result_dir + f'/inventory_decisions/{station_id}_inventory_decision_best_possible.npy', best_possible_decision)
    return inventory_decision_dict


# (3) Evaluation:
def getAllDaysExpectedCosts(_pickupRates, _returnRates, _inventory_decision, _C, _date_list, _lengthT, _t, _quarter):
    Costs_month = []
    for dt in range(len(_date_list)):
        cost = calExpectedCost(_pickupRates[dt], _returnRates[dt], _C, _inventory_decision[dt], _lengthT, _t, _quarter)
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

# Wrapper function   
def get_inventory_decision_evaluation_results(station_id, date_list, hour_range, quarter, model_type, flag_benchmark, data_dir='data', result_dir='saved_files'):
    #define the number of aggregations
    quarter=range(int(60/quarter))
    _lengthT = len(hour_range)*len(quarter)
    #read station data
    capacity, df_booking, df_demand = readStationData(station_id, quarter, data_dir)

    #read inventory decisions
    inventory_decision_dict = {}
    inventory_decision_dir = f'{result_dir}/inventory_decisions'
    for key in model_type:
        inventory_decision_dict[key] = np.load(f'{inventory_decision_dir}/{station_id}_inventory_decision_{key}.npy')
    if flag_benchmark:
        inventory_decision_dict['poisson_rnn'] = np.load(f'{inventory_decision_dir}/{station_id}_inventory_decision_poisson_rnn.npy')
        inventory_decision_dict['lr'] = np.load(f'{inventory_decision_dir}/{station_id}_inventory_decision_lr.npy')
        inventory_decision_dict['best_possible'] = np.load(f'{inventory_decision_dir}/{station_id}_inventory_decision_best_possible.npy')
        inventory_decision_dict['true_count'] = np.load(f'{inventory_decision_dir}/{station_id}_inventory_decision_true_count.npy')
        inventory_decision_dict['overall_ha'] = np.load(f'{inventory_decision_dir}/{station_id}_inventory_decision_overall_ha.npy')
        inventory_decision_dict['specific_ha'] = np.load(f'{inventory_decision_dir}/{station_id}_inventory_decision_specific_ha.npy')

    for key in inventory_decision_dict.keys():
        inventory_decision_dict[key] = list(map(int,inventory_decision_dict[key]))
        print(key, inventory_decision_dict[key])
    # calculate expected costs
    print("\nEvaluation starts ...")
    pickup_ture_count_list = []
    return_ture_count_list = []    
    for dt in date_list:
        pickup_ture_count,return_ture_count = getTrueRate(df_demand, dt, hour_range, quarter)
        pickup_ture_count_list.append(pickup_ture_count)
        return_ture_count_list.append(return_ture_count)
    expected_cost_dict = {}
    for key,val in inventory_decision_dict.items():
        if key != 'best_possible':
            expected_cost_dict[key] = \
                getAllDaysExpectedCosts(pickup_ture_count_list, return_ture_count_list,
                val, capacity, date_list, _lengthT, _t=1, _quarter=quarter)
            np.save(result_dir + f'/results/optimization/expected_costs/{station_id}_expected_cost_' + key + '.npy', expected_cost_dict[key])
            
    # calculate real costs
    real_cost_dict = {}
    for key,val in inventory_decision_dict.items():
        real_cost_dict[key] = getAllDaysRealCosts(val, capacity, df_booking, date_list, hour_range)
        np.save(result_dir + f'/results/optimization/real_costs/{station_id}_real_cost_' + key + '.npy', real_cost_dict[key])
    print("\nEvaluation finished! Results are saved in saved_files/results/optimization/")

    #output
    with open(result_dir + f"/results/optimization/summary/output_st{station_id}.txt", "w") as text_file:
        print(f" STATION {station_id} - expected costs", file=text_file)
        print("---------------\n", file=text_file)
        for key in expected_cost_dict.keys():
            mean = np.array(expected_cost_dict[key]).mean()
            print("| " + key + f" |  MEAN: {mean:.2f}\n", file=text_file)
        print(f" STATION {station_id} - real costs", file=text_file)
        print("---------------\n", file=text_file)
        for key in real_cost_dict.keys():
            mean = np.array(real_cost_dict[key]).mean()
            print("| " + key + f" |  MEAN: {mean:.2f}\n", file=text_file)