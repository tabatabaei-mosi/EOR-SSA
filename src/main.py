import time
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from hybrid_model import SSA_ANN
from preprocessor import preprocessor
from utils import (conf_matrix, history_process, log_and_print, plot_metrics,
                   write_df)

# Ignore warnings
warnings.filterwarnings("ignore")

# Start measuring time
start_time = time.time()

# Set plot font family
plt.rcParams.update({'font.family': 'Times New Roman'})

# Data splitting fractions
train_frac = 0.70
test_frac = 0.15
val_frac = 0.15

# Neural network architecture
hidden_nodes = [52, 45, 38, 31, 24, 17, 10]
activation = 'tanh'  # Activation function for all hidden layers

# Hyper-parameters for SSA
epochs = 2
population_size = 20
producers_frac = 0.8
scrounger_frac = 0.2
safety_threshold = 0.5  # Range [0.5, 1]

# Multi-objective settings
multi_obj_func = False
loss_function = 'CCE'  # Can be 'CCE' or 'RMSE'

# Random seed for reproducibility
random_seed = 42

# Read the dataset
data_file = pd.read_excel("data/EOR_SSA(V3).xlsx", sheet_name='EOR_class_data')

# convert the dataset to numpy array
data_np = data_file.to_numpy()

# get the number of input features
n_feature = data_np.shape[1] - 1

# preprocess the dataset
dataset = preprocessor(
    data_np,
    inp_feature=n_feature,
    test_frac=test_frac,
    normalize_data=True,
    seed=random_seed
)

# Optimize and create the EOR_model using SSA
EOR_model = SSA_ANN(
    dataset,
    hidden_nodes,
    activation_list=[activation * len(hidden_nodes)],
    epoch=epochs,
    population_size=population_size,
    ST=safety_threshold,
    Producers=producers_frac, Scrounger=scrounger_frac,
)

# history call
train_loss_history, train_solution_history, train_gbf_history = EOR_model.training()

# training runtime
train_run_time = time.time() - start_time

# Get the history of DNN for train, test, and validation datasets
train_m_h = history_process(
    xdata=dataset['x_train'], ydata=dataset['y_train'],
    n_epoch=epochs,
    weights_history=train_solution_history,
    ann_model=EOR_model
)

test_m_h = history_process(
    xdata=dataset['x_test'], ydata=dataset['y_test'],
    n_epoch=epochs,
    weights_history=train_solution_history,
    ann_model=EOR_model
)


train_cce_h = history_process(
    xdata=dataset['x_train'], ydata=dataset['y_train'],
    n_epoch=epochs,
    weights_history=train_solution_history,
    ann_model=EOR_model,
    cross_entropy_loss=True
)

test_cce_h = history_process(
    xdata=dataset['x_test'], ydata=dataset['y_test'],
    n_epoch=epochs,
    weights_history=train_solution_history,
    ann_model=EOR_model,
    cross_entropy_loss=True
)

# Calculate confusion matrices for train, test datasets
train_cm, train_label = conf_matrix(
    xdata=dataset['x_train'], ydata=dataset['y_train'],
    n_epoch=epochs,
    ann_model=EOR_model,
    weights_history=train_solution_history,
    Category=True
)

test_cm, test_label = conf_matrix(
    xdata=dataset['x_test'], ydata=dataset['y_test'],
    n_epoch=epochs,
    ann_model=EOR_model,
    weights_history=train_solution_history,
    Category=True
)


# Extract metrics for the last epoch
train_cce_le, test_cce_le = train_m_h['cce_history'][-1], test_m_h['cce_history'][-1]
train_rc_le, test_rc_le = train_m_h['recall_history'][-1], test_m_h['recall_history'][-1]
train_pr_le, test_pr_le = train_m_h['precision_history'][-1], test_m_h['precision_history'][-1]
train_f1_le, test_f1_le = train_m_h['f1_history'][-1], test_m_h['f1_history'][-1]


history_time = time.time()


# Create a dictionary of metrics
metric_dic = {'CCE Loss': [train_cce_h[-1], test_cce_h[-1]],
              'CCE(%)': [train_cce_le, test_cce_le],
              'Recall(%)': [train_rc_le, test_rc_le],
              'Precision(%)': [train_pr_le, test_pr_le],
              'F1 score(%)': [train_f1_le, test_f1_le]
              }

metric_df_le = pd.DataFrame(metric_dic, index=['Train', 'Test', 'Validation'])

log_and_print(
    f"This is the result of Sparrow search algorithm for optimizing DNN's weight and bias --> V.6.0")
log_and_print(
    f"The result is for last epoch, i.e. best solution of training process.")

# Log the metric results
logger.info(
    "This is the result of Sparrow search algorithm for optimizing DNN's weight and bias --> V.6.0")
logger.info(
    "The result is for the last epoch, i.e., the best solution of the training process.")
logger.info(metric_df_le)

logger.info(
    f'>The hidden layer information ----> {hidden_nodes} \n>The Activation list ----> {activation} \n>The SSA parameters:')
logger.info(
    f'>PD={producers_frac}, SD={scrounger_frac}, ST={safety_threshold}, pop_size={population_size}')
logger.info(
    f'>Multi-objective function : {multi_obj_func} -----> The loss function was : {loss_function}')
logger.info(
    f'[#]. Train process run time : {round(train_run_time - start_time, 3)} sec ----> {round(((train_run_time - start_time) / 60), 3)} min')
logger.info(
    f'[##]. History process run time : {round(history_time - train_run_time, 3)} sec ----> {round(((history_time - train_run_time) / 60), 3)} min')

# Calculate the overall program run time
end = time.time()
time_proceess = round((end - start_time), 3)
logger.info(
    f"The Overall Program Run time was {time_proceess} sec ----> {round((time_proceess / 60), 3)} min")


plot_metrics(
    train_metrics=train_cce_h,
    test_metrics=test_cce_h,
    train_cm=train_cm,
    test_cm=test_cm,
    epochs=epochs,
    label=train_label
)
