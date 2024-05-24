import matplotlib.pyplot as plt
import numpy as np

from Helper import smooth

#https://medium.com/@aiblogtech/what-is-exploration-strategies-in-reinforcement-learning-32677239245e

def run_repetitions_bolztman():
    print()

def experiment_boltzman():
    for i in range(10):
        some_curve = run_repetitions_bolztman()

        avg_curve = smooth(some_curve, window=20, )

        plt.plot(avg_curve, label=f'boltzman={i}')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.legend()
    plt.title('Performance of boltzman')
    plt.savefig('fig_name.png')
    plt.show()
def experiment():
    experiment_boltzman()


if __name__ == '__main__':
    experiment()