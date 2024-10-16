

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
    regression_line = beta * years + alfa
    regression_line_nonlog = beta_nonlog * years + alfa_nonlog 

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


    #2f)
    #by reading graph from x,y = 2018, 9  -> 2019.4, 10
    # we see a factor 10 increase which gives 1.4 years
    
  
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

if __name__ == "__main__":
    main()

#----------------- 3a -----------------------------

# Använder formeln för T-testet vilket ger t = 0.8/(σ (0.8)) ≈ 0.8/0.414 = 1.93

# degrees of freedom, 6-2 = 4 : α = 0.05

# Jämför vårt värde med t-tabell och får 2.776, eftersom vårat värde är mindre kan inte en linjär

# signifikant relation bevisas.

 

#----------------- 3b -----------------------------

# konfidensintervall ges av 0.8 +- 2.776 * 0.414 vilket ger (-0.35,1.95)

# Något som visar på att lutningen kan variera mellan -0.35 och 1.95. Eftersom

# lutningen innefattar 0, kan inte med säkerhet säga att lutningen är positiv. Detta

# medför en osökerhet att svara på frågan med detta konfidensintervall.

 

 

#----------------- 3c -----------------------------

# För 2025 kan vi förvänta oss:

# y = 0.8 * 2025 - 1604.56 = 6.44 | För att beräkna konfidensintervall för 2025:

# konfidensintervall = 6.44 +- 2.776 * 1.593 = (1.00,11.88)

# konfidensintervall är så pass stort vilket medför en stor osäkerhet

 

#----------------- 3d -----------------------------

# 0.80*2030−1604.56=8.44

# konfidensintervall 8.44 +- 2.776 * .593 = (3.00,13.88)

#

# Slutsats likt 3c

