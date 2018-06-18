import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
import pandas as pd




def plot_results_from_was_correct_prediction():
    # F1 score of user credibility. Plot of increasing N (how many tweets on topic improve prediction)
    data = [0.5984916, 0.6653914, 0.66831789, 0.68621235, 0.65942141, 0.67253662, 0.66927723, 0.65649999, 0.73633677, 0.65481319, 0.73562632, 0.64899688, 0.66558473, 0.7149242, 0.72146499]
    plt.title('Credibility Prediction')

    plt.plot(range(len(data)), data, 'b', label='Prediction with increasing N.')
    plt.legend(loc='lower right')
    plt.xlim([-0.1,15])
    plt.ylim([-0.1,1.01])
    plt.ylabel('F1 %')
    plt.show()

def plot_results_from_was_correct_prection_with_adding_useful_tokens_to_fact_representation():
    # F1 score of user credibility. Plot of increasing N (how many tweets on topic improve prediction)
    # Add useful tokens of a tweet to the fact representation after computing the distance
    data = [0.53951037,0.63567741,0.65003746,0.72301152,0.70285412,0.69406246,0.637994,0.63524401,0.7020219 ,0.61444726,0.66037611,0.62539298,0.69111688,0.66434283,0.68754011]
    ax = sns.tsplot(data=data)
    ax.set(xlabel='Credibility prediction with n data points', ylabel='F1 Score')
    plt.show()

    # plt.title('Credibility Prediction')
    # plt.plot(range(len(data)), data, 'b', label='Prediction with increasing N.')
    # plt.legend(loc='lower right')
    # plt.xlim([-0.1,15])
    # plt.ylim([-0.1,1.01])
    # plt.ylabel('F1 %')
    # plt.show()

def plot_results_lstm_early():
    # F1 score of user credibility. Plot of increasing N (how many tweets on topic improve prediction)
    # Add useful tokens of a tweet to the fact representation after computing the distance
    data = [0.5818,0.7509, 0.7317, 0.746, 0.8622, 0.8284, 0.8144, 0.8989, 0.8501, 0.9467]
    n = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    df = pd.DataFrame({'data': data, 'n': n})

    ax = sns.tsplot(data=data, time=n)
    ax.set(xlabel='Credibility prediction with n data points', ylabel='F1 Score')
    plt.show()

#plot_results_from_was_correct_prediction()
#plot_results_from_was_correct_prection_with_adding_useful_tokens_to_fact_representation()
plot_results_lstm_early()