import matplotlib.pyplot as plt




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


plot_results_from_was_correct_prediction()