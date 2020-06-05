# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import pickle
import time


#Note: Try avoiding large values on money related parameters. Better count in terms of millions of dollars to avoid numerical problems when running the game. Will resolve this on a latter version.
d     = 1.15
gamma = .2 #Note: r-gamma/10 must be a number between 0 and 1 when running the game bellow
g     = .4
cost  = 7
max_t = 10
mu    = 0
sigma = .75 #Note if you set this to 0 there is no randomness and you can see the fundamentals
k     = 4
h     = .75

p          = .15  #Chance the company booms
r          = .07  #Discount rate
avg_rate   = .015 #Average investment return rate 
sigma_r    = .005 #Investment return rate variance
initial_ar = 12   #Initial available resources the startup has

version_name = "10_to_30"
model_path   = version_name + "/model"
results_path = version_name + "/results"
figures_path = version_name + "/figures"

os.mkdir(version_name)
os.mkdir(model_path)
os.mkdir(results_path)
os.mkdir(figures_path)

def profit_gen(d, gamma, g, cost, t, sigma, st_type, k = None, h = None):
  #Function to generate yearly profits
  #inputs:
    #d:       number related to revenue at the start  of the game (float)
    #gamma:   number related the exponential growth of revenue (float)
    #g:       number related to cost growth (float)
    #cost:    number related to fixed yearly cost (float)
    #t:       year (integer)
    #sigma:   variance of the normal random variable (float)
    #st_type: type of startup (string, available entries ['boom', 'flop'])
    #k:       number of years both companies behave the same (integer)
    #h:       number related to the linear growth of revenue for the flop (float)
  #output:
    #profit:  company's profit for that year (float)
  xi =  np.random.normal(0, sigma, 1)[0]
  if st_type == 'boom':
    profit = np.exp(gamma*t + d) - g*t - cost + xi
  elif st_type == 'flop':
    profit = h*(t - k) + np.exp(gamma*k + d) - g*t - cost + xi
  return profit

# def path_simulation(d, gamma, g, cost, max_t, sigma, k, h):
  # #Function to simulate a sequence of profits for both types of startups
  # ps_b = []
  # ps_f = []
  # for t in range(1, max_t + 1):
    # if t <= k:
      # ps_b += [profit_gen(d, gamma, g, cost, t, sigma, 'boom')]
      # ps_f += [profit_gen(d, gamma, g, cost, t, sigma, 'boom')]
    # else:
      # ps_b += [profit_gen(d, gamma, g, cost, t, sigma, 'boom')]
      # ps_f += [profit_gen(d, gamma, g, cost, t, sigma, 'flop', k, h)]
  # return ps_b, ps_f

# def profit_plot(d, gamma, g, cost, max_t, sigma, k, h):
  # #Function to plot a sequence of profits for both types of startups
  # plt.figure(figsize = (10, 10))
  # plt.title('Startups profits', size = 40)
  # ps = path_simulation(d, gamma, g, cost, max_t, sigma, k, h)
  # plt.plot(range(1, max_t + 1), ps[0], label = 'Boom', color = 'blue')
  # plt.plot(range(1, max_t + 1), ps[1], label = 'Flop', color = 'green')
  # plt.plot(range(1, max_t + 1), [0]*max_t, color = 'red')
  # plt.xlabel("Year", size = 25)
  # plt.ylabel("Millions of Dollars", size = 25)
  # plt.legend(prop = {'size': 20})
  # plt.show()



#profit_plot(d, gamma, g, cost, max_t, sigma, k, h)


def t_bill_rate(avg_rate, sigma_r):
  #Function to generate a given year's return on their investment. We consider a minimum non-zero rate. Too small rates might cause numerical problems.
  #Inputs:
    #avg_rate: The average yearly rate (float)
    #sigma_r:  The variance of the yearly rate (float)
    #min_r:    The minimum interest rate (rate)
  #Output:
    #tbr:      The given year's rate (float)
  tbr =  np.random.normal(0, sigma_r, 1)[0]
  return tbr


def transition_func(prop, avail_res, t, st_type):
  #Function to determine future available resources given present action and available resources
  #Inputs:
    #prop:        Proportion of available resources distributed by the startup. (float, available entries [.0, .1, .2, .3] )
    #avail_res:   Amount of available resources (float)
    #t:           Year (integer)
    #st_type:     Type of startup (string, available entries ['boom', 'flop'])
  #Output:
    #company_val: Avaliable resources on the following year
  assert prop in [.0, .1, .2, .3]
  assert type(t) == int
  assert st_type in ['flop', 'boom']
  tbr = t_bill_rate(avg_rate, sigma_r) #Generate the t-bill rate
  #Generate profits
  if st_type == 'flop':
    profit = profit_gen(d, gamma, g, cost, t + 1,sigma, 'flop', k, h)
  elif st_type == 'boom':
    profit = profit_gen(d, gamma, g, cost, t + 1, sigma, 'boom')
  fut_avail_res = (1 + tbr) * (1 - prop) * avail_res + profit
  return fut_avail_res, profit

def final_value(avail_res, r, max_t, st_type, gamma = None, d = None, h = None, g = None):
  #Function to the determine the company's value once bought if it lasts till the end of the game. If the value is negative company will yield a reward of 0 instead
  #Inputs:
    #[avail_res, st_type, gamma, d, h, g]: Already defined    
    #r:           Discount rate. (float we need (r - gamma/10) to be between 0 and 1)
    #max_t:       Number of years until company gets bought (integer)
  #Outputs:
    #company_val: The company's valuation once it gets bought/liquidated on the last year (float)
  assert st_type in ['flop', 'boom']
  assert 0 < r - .2 * gamma < 1
  assert type(max_t) == int
  if st_type == 'boom':
    future_profits_pv = (1 / (r - .2 * gamma)) * np.exp(d - (r -.2 * gamma) * max_t)
  elif st_type == 'flop':
    future_profits_pv = (.2 * (h - g) * max_t + .2 * (h - g)/r) * np.exp(-r * max_t)/r
  company_val = avail_res + future_profits_pv
  company_val = np.max([company_val, 0])
  return company_val

def step(prop, avail_res, t, st_type):
  #Function to determine what happens after a given action is taken
  #Inputs:
    #[prop, avail_res, t, st_type]: Already defined
  #Outputs:
    #fut_avail_res: Amount of resources that will be available on the following year (float())
    #reward:        How much utility or "hapiness" the given action generates that year (float)
    #done:          Boolean to inform if the game is over (boolean)
    #fut_profit:        Profit for the following year
    #company_val:   How much the company was sold for (float)
  assert prop in [.0, .1, .2, .3]
  assert type(t) == int
  assert st_type in ['flop', 'boom']
  fut_avail_res, next_profit = transition_func(prop, avail_res, t, st_type)
  if t == max_t:
    done = True
    future_profits = final_value(fut_avail_res, r, max_t + 1, st_type, gamma, d , h, g)
    company_val    = np.max([fut_avail_res + future_profits * (1/(1 + r)), 0])
    reward         = np.log(1 + prop * avail_res + company_val)
    return fut_avail_res, reward, done, next_profit, company_val
  elif fut_avail_res <= 0:
    done   = True
    reward = np.log(1 + prop * avail_res)
    return fut_avail_res, reward, done, next_profit, 0
  else:
    done   = False
    reward = np.log(1 + prop * avail_res)
    return fut_avail_res, reward, done, next_profit, 0




def generate_model(layer_arquitecture, activation, drop_rate, out_shape, loss, optimizer):
  #Funtion to generate a neural network model on tensorflow
  #Inputs:
    #layer_arquitecture: Sequence with the number of neurons on each layer. (list of integers)
    #activation:         The activation function for each intermediary layer (string)
    #drop_rate:          The percentage of random neurons nullified on each iteration to decrease overfitting (float between 0 and 1)
    #out_shape:          Number of possible actions (integer)
    #loss:               Loss function used to train the model (string)
    #optimizer:          The algorithm used for optimization (string)
  #Outputs:
    #model:              A compiled model using a sigmoid function on the last layer (tensorflow model instance)
  layers = [tf.keras.layers.InputLayer(input_dim = 2)]
  for val in layer_arquitecture:
    layers += [tf.keras.layers.Dense(val, activation = activation)]
  layers += [tf.keras.layers.Dense(out_shape, activation = 'sigmoid')]
  model   = tf.keras.models.Sequential(layers)
  model.compile(loss = loss, optimizer = optimizer, metrics = ['mse', 'mae'])
  return model


try:
  model = tf.keras.models.load_model(model_path + '/model')
  new_model = False
except:
  model = generate_model([8, 8], 'relu', .15, 4, 'mse', 'adam')
  new_model = True



#In case you want to plot a previous result uncomment this:
try:
  with open(results_path  + 'avg_ret.pickle', 'rb') as handle:
    avg_ret = pickle.load(handle)
except:
  avg_ret = {'boom' : {'Utility' : [], 'Valuation' : [], 'Available Resources' : [], 'Duration' : [], 'Average proportion' : [], 'Profit History' : []}, 
            'flop' : {'Utility' : [], 'Valuation' : [], 'Available Resources' : [], 'Duration' : [], 'Average proportion' : [], 'Profit History' : []},
            'Full' : {'Utility' : [], 'Valuation' : [], 'Available Resources' : [], 'Duration' : [], 'Average proportion' : [], 'Profit History' : []}}



def game(model, eps, initial_avail_res, lr, p):
  #Function that runs the whole startup game once
  #Inputs:
    #model:   The model instance to train the bot (model instance)
    #eps:     The amount of exploration the bot tries different things during this game (float between 0 and 1)
    #lr:      Learning rate for the bot (float between 0 and 1)
    #p:       The chance the company will be a booming startup after date k (float between 0 and 1)
  #Outputs:
    #model:   The updated model instace after the game (model instance)
    #r_sum:   How much reward the bot got this game (float)
    #st_type: The type of startup it ran this game (string, available entries ['boom', 'flop'])
  r_sum = 0
  done = False
  t = 1
  st_type = 'boom'
  avail_res = initial_avail_res
  action_history = []
  profit_history = []
  while not done:
    if np.random.uniform() < eps:
      prop = np.random.randint(0, 4) / 10
    else:
      prop = np.argmax(model.predict([[avail_res, t]])) / 10
    fut_avail_res, reward, done, next_profit, company_val = step(prop, avail_res, t, st_type)
    action_history    += [prop]
    profit_history    += [next_profit]
    full_state         = np.array([fut_avail_res, t + 1]).reshape(1, 2)
    target             = reward + (1/(1 + r)) * np.max(model.predict([[full_state]]))
    target_vec         = model.predict([[fut_avail_res, t + 1]])
    pos                = int(prop*10)
    target_vec[0][pos] = (1 - lr) *target_vec[0][pos] + lr * target
    x_temp             = np.array([avail_res, t]).reshape(1, 2)
    y_temp             = np.array(target_vec).reshape(1, 4)
    model.fit(x_temp, y_temp, verbose = 0)
    avail_res          = fut_avail_res
    t                 += 1
    if t == k + 1 and np.random.uniform() >= p:
      st_type = 'flop'
    r_sum             += reward
  mean_prop            = np.mean(action_history)
  return model, r_sum, st_type, fut_avail_res, company_val, t, mean_prop, profit_history



# if new_model == False:
  # new_model_save_name = input("What should be the filename for the new model? ")
  # new_result_save_name = input("What should be the result filename for the new model? ")

if new_model == True:
  num_runs = 100000
  decay    = .999999
  eps      = .5
else:
  num_runs = 25000
  decay    = .99999
  eps      = .25

lr         = .95

start_time = time.time()

for i in range(num_runs):
  eps *= decay
  if i % 500 == 0:
  print("Episode {} of {}".format(i + 1, num_runs))
  # if new_model == True:
  model.save(model_path + '/model')
  with open(results_path  + 'avg_ret.pickle', 'wb') as handle:
    pickle.dump(avg_ret, handle)
        # else:
          # model.save('gdrive/My Drive/Colab Notebooks/Results/' + new_model_save_name)
          # with open('gdrive/My Drive/Colab Notebooks/Results/' + new_result_save_name + '.pickle', 'wb') as handle:
            # pickle.dump(avg_ret, handle)
  model, r_sum, st_type, fut_avail_res, company_val, t, mean_prop, profit_history = game(model, eps, initial_ar, lr, p)

  avg_ret[st_type]['Utility']             += [r_sum]
  avg_ret[st_type]['Available Resources'] += [fut_avail_res]
  avg_ret[st_type]['Valuation']           += [company_val]
  avg_ret[st_type]['Duration']            += [t]
  avg_ret[st_type]['Average proportion']  += [mean_prop]
  avg_ret[st_type]['Profit History']      += [profit_history]

  avg_ret['Full']['Utility']              += [r_sum]
  avg_ret['Full']['Available Resources']  += [fut_avail_res]
  avg_ret['Full']['Valuation']            += [company_val]
  avg_ret['Full']['Duration']             += [t]
  avg_ret['Full']['Average proportion']   += [mean_prop]
  avg_ret['Full']['Profit History']       += [profit_history]

training_duration = (time.time() - start_time)
if training_duration >= 60:
  print('\n \n Training took {} minutes'.format(training_duration/60))
else:
  print('\n \n Training took {} seconds'.format(training_duration))

# if new_model == True:
model.save(model_path + '/model')
# else:
  # model.save('gdrive/My Drive/Colab Notebooks/Results/' + new_model_save_name)

# if new_model == True:
with open(results_path  + 'avg_ret.pickle', 'wb') as handle:
  pickle.dump(avg_ret, handle)
# else:
  # with open('gdrive/My Drive/Colab Notebooks/Results/' + new_result_save_name + '.pickle', 'wb') as handle: #needs pythonization
    # pickle.dump(avg_ret, handle)

def moving_average(result_seq, size):
  N = len(result_seq)
  rolling_mean = []
  for i in range(size, N+ 1):
    rolling_mean += [np.mean(result_seq[i - size: i])*100]
  return rolling_mean

def sold_boolean(valuations):
  indicator = np.array(valuations) > 0
  return indicator

def boom_long_term_startup(duration_vals, list_vals):
  indexes = [ind for ind in range(len(duration_vals)) if duration_vals[ind] <= k]
  short_lived = [list_vals[ind] for ind in indexes]
  long_lived = [list_vals[ind] for ind in range(len(duration_vals)) if ind not in indexes]
  return short_lived, long_lived

def bot_evolution_plot(avg_ret, size):
  #Function to plot the bot evolution in terms of total reward, company valuation, company duration, and average cash distribution each game
  for key in avg_ret['boom'].keys():
    if key == 'Profit History':
      continue
    short_lived, long_lived = boom_long_term_startup(avg_ret['boom']['Duration'], avg_ret['boom'][key])
    try:

      long_lived = moving_average(long_lived, size)
      flop_vals  = moving_average(avg_ret['flop'][key], size)
      full_vals  = moving_average(avg_ret['Full'][key], size)
      fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (12, 12), sharey = True)
      plt.suptitle(key, size = 40)    
      

      ax1.plot(flop_vals, label = 'Flop', color = 'green')
      ax2.plot(long_lived, label = 'Boom', color = 'blue')
      ax3.plot(full_vals, label = 'Complete Run', color = 'red')

      ax1.set_xlabel("Iteration", size = 25)
      ax2.set_xlabel("Iteration", size = 25)
      ax3.set_xlabel("Iteration", size = 25)

      if key == 'Utility':
        ax1.set_ylabel("Logarithmic Scale", size = 25)
      elif key == 'Valuation':
        ax1.set_ylabel("Available Resources", size = 25)
      elif key == 'Valuation':
        ax1.set_ylabel("Millions of Dollars", size = 25)
      elif key == 'Duration':
        ax1.set_ylabel("Number of Years", size = 25)
      elif key == 'Average proportion':
        ax1.set_ylabel("Percentage", size = 25)

      ax1.legend(prop = {'size': 20})
      ax2.legend(prop = {'size': 20})
      ax3.legend(prop = {'size': 20})
      plt.show()   
      plt.savefig(figures_path + '/'+ key + '.png')
    except Exception as ex:
      print(ex)
      print(key)

bot_evolution_plot(avg_ret, 100)

def sold_proportion(avg_ret, size):
  boom_prop = moving_average(sold_boolean(avg_ret['boom']['Valuation']), size)
  flop_prop = moving_average(sold_boolean(avg_ret['flop']['Valuation']), size)
  full_prop = moving_average(sold_boolean(avg_ret['Full']['Valuation']), size)
  fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (12, 12), sharey = True)
  plt.suptitle('Proportion of Sold Companies', size = 40) 

  ax1.plot(flop_prop, label = 'Flop', color = 'green')
  ax2.plot(boom_prop, label = 'Boom', color = 'blue')
  ax3.plot(full_prop, label = 'Complete run', color = 'red')

  ax1.set_xlabel("Iteration", size = 25)
  ax2.set_xlabel("Iteration", size = 25)
  ax3.set_xlabel("Iteration", size = 25)
  ax1.set_ylabel("Proportion", size = 25)
  plt.legend(prop = {'size': 20})
  plt.show()    
  plt.savefig(figures_path + '/Sucess_rate' + '.png')

sold_proportion(avg_ret, 100)

