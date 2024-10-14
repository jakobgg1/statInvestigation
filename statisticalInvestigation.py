import matplotlib.pyplot as plt
import math
def data():
    elmo = [2018, 94_000_000]
    gpt2 = [2019, 1_500_000_000]
    megatron_lm = [2019, 8_300_000_000]
    turing = [2020, 17_200_000_000]
    gpt3 = [2020, 175_000_000_000]
    gpt4 = [2023, 1_800_000_000_000]
    
    models = [elmo, gpt2, megatron_lm, turing, gpt3, gpt4]
    
#https://medium.com/@haytamborquane/nvidia-megatron-lm-model-ae9c91794eec
#for gpt4: https://explodingtopics.com/blog/gpt-parameters
    return models

def plot(data):
    years = []
    parameters = []
    log_parameters = []
    for e in data:
        years.append(e[0])
        parameters.append(e[1])
        log_parameters.append(math.log10(e[1]))

    a, b = regression_estimation(log_parameters) #excercise 2


    fig, ax1 = plt.subplots()

    ax1.scatter(years, parameters, c='blue') #b)
    ax1.set_title('Non-logarithmic')
    plt.show(block=False)

    fig2, ax2 = plt.subplots()

    ax2.scatter(years, log_parameters, c='red') #c-d)
    ax2.set_title('logaritm')
    plt.show()

def regression_estimation(y):

    pass



def main():
    plot(data())


main()

#answers 1e):
#seems to be growing exponantially, not linear


#---------------------2 ----------------------------
#a = initial, b = growth rate, x year, c deviations.
#https://ourworldindata.org/moores-law 
#historically nr of transistors per computer chip has doubled.
