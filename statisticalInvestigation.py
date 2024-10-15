

import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.linalg import lstsq

#---------------------1 ----------------------------
def data():
    elmo = [2018, 94_000_000]
    gpt2 = [2019, 1_500_000_000]
    megatron_lm = [2019, 8_300_000_000]
    turing = [2020, 17_200_000_000]
    gpt3 = [2020, 175_000_000_000]
    gpt4 = [2023, 1_800_000_000_000]
    
    models = [elmo, gpt2, megatron_lm, turing, gpt3, gpt4]
    return models

# 1a   
#   
#   https://medium.com/@haytamborquane/nvidia-megatron-lm-model-ae9c91794eec
#   for gpt4: https://explodingtopics.com/blog/gpt-parameters
    

2
def plot(data):
    years = []
    parameters = []
    log_parameters = []
    for e in data:
        years.append(e[0])
        parameters.append(e[1])
        log_parameters.append(math.log10(e[1]))

    alfa, beta = linearLeastSqueres(years, log_parameters) #excercise 2
    alfa_nonlog, beta_nonlog = linearLeastSqueres(years, parameters)
    '''  #ignore
    for i in range(len(years)): 
        temp = beta * data[i][0] + alfa
        log_val = math.log10(data[i][1])
        diff = (log_val-temp)
        print(data[i][0], '=', diff, 'expected linearvalue:', temp) #2d)
    '''
    years = np.array(years)
    regression_line = (beta) * years + (alfa)
    regression_line_nonlog = beta_nonlog * years + alfa_nonlog 
    #parameters = np.array(parameters)

    
    fig, ax1 = plt.subplots()

    ax1.scatter(years, parameters, c='blue') #b)
    ax1.set_title('Non-logarithmic')
    ax1.plot(years, regression_line_nonlog, c='red')
    plt.show(block=False)


    fig2, ax2 = plt.subplots()
    ax2.scatter(years, log_parameters, c='red') #c-d)
    ax2.plot(years, regression_line, 'r', label='Fitted line') #2e)

    ax2.set_title('logaritm')
    plt.show()

    #residual line:
  
#answers 1e):
#seems to be growing exponantially, not linearly
#---------------------2 ----------------------------
#2a 
#   a = initial, b = growth rate, x year, c deviations.
#   Morse law - historically nr of transistors per computer chip doubles every two years. (https://ourworldindata.org/moores-law)
    
#2b&c
def linearLeastSqueres(x, y):
    x = np.array(x)
    y = np.array(y)
    A = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, y)[0]
    return b, a

#2c



def main():
    plot(data())
    #linearLeastSqueres(x, y)

if __name__ == "__main__":
    main()
