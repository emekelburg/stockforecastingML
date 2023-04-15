import time
import pandas as pd
import numpy as np
import random
import argparse

from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBRegressor

#from optuna.integration import KerasPruningCallback
#from optuna.trial import TrialState

from sklearn.linear_model import ElasticNet

import keras
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LeakyReLU

from lightgbm import LGBMRegressor

import optuna

#import plotly
#import plotly.io as pio
#pio.renderers.default = "colab"

import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # NN training causes a lot of these



#######

def ts_split (i_dataset, i_initial_training_window_size = 120, i_assessment_window_size=1, i_gap=0, i_cumulative=False):

  splits = []
  n_obs = len(i_dataset) 

  offset = 0
  while(offset <= n_obs - i_initial_training_window_size - i_assessment_window_size - i_gap):
    if(i_cumulative == False):
      
      split_ = pd.concat( [
            # append training data set
            i_dataset.iloc[ offset : offset+i_initial_training_window_size ,] ,
            # assessment data set
            i_dataset.iloc[ offset+i_initial_training_window_size+i_gap : offset+i_initial_training_window_size+i_gap+i_assessment_window_size, ]
            ])
      splits.append(split_)
    else:

      #splits.append ( i_dataset.iloc[ 0 : offset+i_initial_training_window_size+i_assessment_window_size ,] )
      split_ = pd.concat( [
      # append training data set
      i_dataset.iloc[ 0 : offset+i_initial_training_window_size ,] , # always start at zero
      # assessment data set
      i_dataset.iloc[ offset+i_initial_training_window_size+i_gap : offset+i_initial_training_window_size+i_gap+i_assessment_window_size, ]
      ])
      splits.append(split_)

    offset+=1

  return splits


def create_model_enet(trial):

  # The parameter l1_ratio corresponds to alpha in the glmnet R package 
  #    (elastic net mixing parameter with range 0..1)
  #  l1_ratio = 1 is the lasso penalty, l1_ratio = 0 is ridge
  l1_ratio_ = trial.suggest_float('l1_ratio', 0, 1, log=False)

  # alpha corresponds to the lambda parameter in glmnet (regularization parameter)
  #alpha_ = trial.suggest_float("alpha", 0.1, 20)
  alpha_ = trial.suggest_float("alpha", 0.01, 20, log=True)

  return ElasticNet(max_iter=10000, alpha=alpha_, l1_ratio=l1_ratio_)

def create_model_lasso(trial):
  #  l1_ratio = 1 is the lasso penalty
  l1_ratio_ = 1
  alpha_ = trial.suggest_float("alpha", 0.01, 20, log=True)
  return ElasticNet(max_iter=10000, alpha=alpha_, l1_ratio=l1_ratio_) 

def create_model_ridge(trial):
  #  l1_ratio = 0 is the ridge penalty
  l1_ratio_ = 0
  alpha_ = trial.suggest_float("alpha", 0.01, 20, log=True)
  return ElasticNet(max_iter=10000, alpha=alpha_, l1_ratio=l1_ratio_)


def create_model_nn_keras(trial):

    # Clear clutter from previous session graphs.
    keras.backend.clear_session()

    # We optimize the number of hidden units per layer and dropout in each layer and
    # the learning rate of RMSProp optimizer.

    # tune number of hidden layers between 1 and 5
    #n_layers = 5 #3 #trial.suggest_int("n_layers", 1, 3)  #fix to three layers
    #max_neurons = [128,64,32,16,8]
    n_layers = trial.suggest_int("n_layers", 1, 5)  #fix to three layers
    max_neurons = [128,64,32,16,8]
    max_neurons[5-n_layers:5]

    activation_function = trial.suggest_categorical("activation_function", ["relu", "leakyRelu"])

    model = Sequential()
    for i in range(n_layers):
        num_hidden = trial.suggest_int("n_units_l{}".format(i), 4, max_neurons[i], log=True)
        if activation_function == "leakyRelu" :
          model.add(Dense(num_hidden, activation=LeakyReLU()))
        else :
          model.add(Dense(num_hidden, activation="relu"))
        dropout = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        model.add(Dropout(rate=dropout))

    model.add(Dense(1))

    # We compile our model with a sampled learning rate.
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    model.compile(
        #loss="mean_absolute_error",
        loss="mean_squared_error",
        #optimizer=RMSprop(learning_rate=learning_rate),
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
        )

    return model

def create_model_NN(trial):
    return MLPRegressor(
              hidden_layer_sizes= trial.suggest_categorical("hidden_layer_sizes", [(32,16,8,4,2),(8,4,2),(100)]),
              activation=         trial.suggest_categorical("activation", ["relu"]), #("activation", ["relu", "identity"]),
              solver=             trial.suggest_categorical("solver", ["adam"]),#("solver", ["sgd", "adam"]),
              learning_rate=      trial.suggest_categorical("learning_rate", ['constant', 'invscaling', 'adaptive']),
              learning_rate_init= trial.suggest_float("learning_rate_init", 0.001, 0.01),
              max_iter =          trial.suggest_int("max_iter", 1000, 1000),
              #early_stopping=True
                      )

def create_model_RF(trial):

  return RandomForestRegressor(
      
        # number of trees in the forest
        n_estimators = trial.suggest_int('n_estimators', 100, 1000), # number of trees

        # minimum number of predictor observations needed to split an internal node
        min_samples_split = trial.suggest_int('min_samples_split', 3, 10),

        # minimum number of target observations in a leaf node
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 3, 10),
        
        # impurity threshold for splitting a node
        #min_impurity_decrease = trial.suggest_float('min_impurity_decrease', 0, 10),

        # maximum depth of each tree
        max_depth = trial.suggest_int('max_depth', 3, 10),

        # maximum number of predictors to consider for a split
        #max_features = trial.suggest_categorical("max_features", ["auto", "sqrt"]),
        max_features = trial.suggest_int('max_features', 5, 20),

        bootstrap = True,
        random_state = 42
      )


  
def create_model_XGB(trial):
    params = {
        "verbosity": 0,  # 0 (silent) - 3 (debug)
        "objective": "reg:squarederror",
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000), #10000,
        "min_child_weight": trial.suggest_float("min_child_weight", 10, 1000),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.6),
        "subsample": trial.suggest_float("subsample", 0.4, 0.8),
        "alpha": trial.suggest_float("alpha", 0.01, 10.0),
        "lambda": trial.suggest_float("lambda", 1e-8, 10.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 10.0),
        "seed": 42,
        "n_jobs": 5,
    }
    return XGBRegressor(**params)

def create_model_LGBM(trial):
    return LGBMRegressor(
              n_estimators = trial.suggest_int("n_estimators", 100, 1000),
              num_leaves = trial.suggest_int('num_leaves', 2, 256),
              reg_alpha = trial.suggest_float('reg_alpha', 1e-8, 10.0), # L1 reg
              reg_lambda = trial.suggest_float('reg_lambda', 1e-8, 10.0), # L2 reg
              learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
              min_child_samples = trial.suggest_int('min_child_samples', 5, 100),
              random_state = 42
                      )

def objective(trial, i_X, i_y, i_model_func):
    # Define model with init values from optuna.
    eval_model = i_model_func(trial) #create_model(trial) 

    allmse = []

    n_splits_ = 5
    n_factor_ = (n_splits_-1)/10

    tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=None)

    df_train_X_baseline = i_X[0:int(len(i_X)*n_factor_)]
    df_train_y_baseline = i_y[0:int(len(i_y)*n_factor_)]

    df_train_X = i_X[int(len(i_X)*n_factor_):len(i_X)]
    df_train_y = i_y[int(len(i_y)*n_factor_):len(i_y)]

    step = 0
    
    # Create datasets in CV scheme considering timeseries data.
    for train_index, test_index in tscv.split(df_train_X):
        X_train, X_test = df_train_X[train_index], df_train_X[test_index]
        Y_train, Y_test = df_train_y[train_index], df_train_y[test_index]

        # merge baseline and CV set together
        X_train = np.concatenate((df_train_X_baseline, X_train))
        Y_train = np.concatenate((df_train_y_baseline, Y_train))

        Y_train = Y_train.ravel()
        Y_test = Y_test.ravel()
        
        # Fit the train data.    
        eval_model.fit(X_train, Y_train)
        
        # Test the model with test data.        
        y_pred = eval_model.predict(X_test)
        
        # Save the mse.
        mse = mean_squared_error(Y_test, y_pred)

        # Report intermediate objective value.
        trial.report(mse, step)
        step += 1

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()      

        allmse.append(mse)
        
    return np.mean(allmse)  # Send mse as feedback to optuna sampler



def tune_ts_model (i_splits, i_model_prefix, i_tune_frequency = 12, i_model_type = "RF", i_trials = 25, i_PCA=False):

  if      i_model_type == "RF": l_model_func = create_model_RF
  elif    i_model_type == "NN": l_model_func = create_model_NN
  elif    i_model_type == "XGB": l_model_func = create_model_XGB
  elif    i_model_type == "NNKeras" : l_model_func = create_model_nn_keras
  elif    i_model_type == "ENET" : l_model_func = create_model_enet
  elif    i_model_type == "LGBM" : l_model_func = create_model_LGBM
  elif    i_model_type == "LASSO" : l_model_func = create_model_lasso
  elif    i_model_type == "RIDGE" : l_model_func = create_model_ridge

  l_model_name_ = i_model_prefix   + "_" + \
                  i_model_type     + "_" + \
                  str(i_tune_frequency)

  n_trials_ = i_trials

  df_output = []
  df_best_params = []

  n_splits = len(i_splits)
  n = 0
  t0 = time.perf_counter()
  #l_current_year = ''
  best_model = object()


  for l_split in i_splits:
  #l_split = splits[1]

    # define standard scaler for x and y variables
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # define zero variance selector
    zv_selector = VarianceThreshold()

    df_input_ = l_split.drop(['y', 'date'], axis=1)

    if i_PCA == True:
      pca = PCA()
      pca.fit(df_input_)
      # find number of components for 99% of variance
      num_comps_ = len(pca.explained_variance_ratio_[np.cumsum(pca.explained_variance_ratio_)<.99])

      pca_act_ = PCA(n_components = num_comps_)
      pca_act_.fit(df_input_)
      df_input_ = pd.DataFrame(pca_act_.transform(df_input_))

    df_scaled_X = df_input_
    df_scaled_X = df_scaled_X.dropna(axis=1) # drop all columns that contain NA values
    df_scaled_X = zv_selector.fit_transform(df_scaled_X) # remove zero variance columns
    df_scaled_X = X_scaler.fit_transform(df_scaled_X) # scale all variables
    df_scaled_train_X = df_scaled_X[0:-1]
    df_scaled_test_X = np.array([df_scaled_X[-1]])

    df_scaled_train_Y = l_split.y.iloc[0:-1]  # remove last row from y-scaling (actual value not avaialble in real time, avoid look ahead bias)
    df_scaled_train_Y = y_scaler.fit_transform(np.array(df_scaled_train_Y).reshape(-1, 1)) # scale the y-values

    # we carry out optimization once per year, within the year, we continue using the same parameters
    #   check if the year has passed

    #if (l_split.iloc[-1].date[:4] != l_current_year or     # check if new year
    #    l_split.iloc[-1].date[5:7] == "01" or              # first month of year
    #    l_split.iloc[-1].date[5:7] == "07" ) :            # seventh month of year
    if (n % i_tune_frequency == 0) :

      print(f'[+] Tuning model for {l_split.iloc[-1].date[:7]}')
      #l_current_year = l_split.iloc[-1].date[:4]

      # trigger parameter optimization     
      study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1), # TPE is optuna default sampler, others cmaes, skopt, etc
                                  direction='minimize',
                                  pruner=optuna.pruners.MedianPruner())

      study.optimize(lambda trial: objective(trial=trial, i_X=df_scaled_train_X, i_y = df_scaled_train_Y, i_model_func=l_model_func), 
                    n_trials=n_trials_, # more is better especially if num param is high and param range is also high
                    catch=(ValueError,)) 
          
      # Show the best params
      best_params = study.best_params
      print(f'[+] best objective value: {study.best_trial.value}')
      print(f'[+] best params: {study.best_params}')

      # create model run output:  date	test.value	mean.train.data	prediction
      best_model = l_model_func(study.best_trial) #create_model(study.best_trial)
    else :
      print(f'[.] Using same tuning parameters for {l_split.iloc[-1].date[:7]}')

    try:    
      # fit model on the whole dataset of this iteration
      best_model.fit(df_scaled_train_X, df_scaled_train_Y)

      # generate prediction for prediction X data
      y_pred = best_model.predict(df_scaled_test_X)

      # inverse transform prediction to actual value based on the scale
      y_pred = y_scaler.inverse_transform( y_pred.reshape(-1, 1) ) [0][0]

    except:
      print("Exception in using same parameters, using historical mean.")
      y_pred = l_split.y.iloc[0:-1].mean()

    df_output.append({'date' : l_split.iloc[-1].date, 
                      'test.value' : l_split.iloc[-1].y, 
                      'mean.train.data': l_split.y.iloc[0:-1].mean(), 
                      'prediction' : y_pred })
    
    n+=1
    print(f'[*] Completed iteration {n}/{n_splits} | elapsed: {time.perf_counter() - t0:0.1f}s | remaining: {(((time.perf_counter() - t0) / n ) * (n_splits - n)):0.1f}')

    df_best_params.append(study.best_params)

  output = pd.DataFrame(df_output)
  output_best_params = pd.DataFrame(df_best_params)

  output.to_csv(l_model_name_+".csv")
  output_best_params.to_csv(l_model_name_+"_best_params.csv")

  return output


parser = argparse.ArgumentParser(description='Time Series Machine Learning Model Generator.')
parser.add_argument('-f', '--filename', type=str, help='filename to be processed', required=True)
parser.add_argument('-p', '--prefix', type=str, help='output file prefix', required=True)
parser.add_argument('-w', '--trainwindow', type=int, help='initial window size', required=True)
parser.add_argument('-t', '--trials', type=int, help='number of trials', required=True)
parser.add_argument('-q', '--tunefreq', type=int, help='frequency of tuning (periods)', required=True)
parser.add_argument('-g', '--gap', type=int, help='gap between training and assessment set (for multi-period returns)', default=0, required=False)
parser.add_argument('-m', '--model', type=str, help='type of model',  required=True,
                          choices=['RF', 'NN', 'XGB', 'NNKeras', 'ENET', 'LGBM','RIDGE','LASSO']),
parser.add_argument('-c','--cumulative', default=False, action="store_true", help='default rolling, set to recursive by activating this flag')
parser.add_argument('-a','--pcaadjust', default=False, action="store_true", help='default no PCA adjustment, set to PCA by setting this flag')

args = parser.parse_args() 

filename_ = args.filename
prefix_ = args.prefix
initial_training_window_size_ = args.trainwindow
cumulative_ = args.cumulative
model_type_ = args.model
trials_ = args.trials
tunefreq_ = args.tunefreq
pcaadjust_ = args.pcaadjust
gap_ = args.gap

output_prefix_ = prefix_ + "_" + str(initial_training_window_size_) + "_" + str(pcaadjust_) + "_"
if cumulative_: output_prefix_=output_prefix_+"recursive"
else: output_prefix_=output_prefix_+"rolling"

df_all = pd.read_csv(filename_)

splits_ = ts_split(     df_all, 
                        i_initial_training_window_size = initial_training_window_size_, 
                        i_assessment_window_size=1, 
                        i_gap=gap_,
                        i_cumulative=cumulative_)

print(f'[*] Number of splits:  {len(splits_)}')

output = tune_ts_model( i_splits=splits_, 
                        i_model_prefix=output_prefix_, 
                        i_tune_frequency = tunefreq_, 
                        i_model_type = model_type_, 
                        i_trials = trials_,
                        i_PCA=pcaadjust_)

#python OOSv12.tuning.py -f="gbr.exp.df.csv" -p="GBR" -w=120 -t=50 -q=100 -m="XGB" 

