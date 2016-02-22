import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    mini = [5.86, 30.13, 103.97]
    alpha = [6.53, 20.17, 68.38]

    mini_w = [0.45, 0.31, 0.68]
    expecti_w = [0.62, 0.67, 0.84]

    data_1 = [mini, alpha]
    for i in xrange(len(data_1)):
        plt.plot(xrange(2, 5), data_1[i])

    plt.xlabel("Depth")
    plt.ylabel("Time (s)")
    plt.title("Time Consumption between \n MiniMax Agent vs. Alpha Beta Agent")
    plt.show()

    data_2 = [mini_w, expecti_w]
    for i in xrange(len(data_2)):
        plt.plot(xrange(2, 5), data_2[i])

    plt.ylim(0, 1.0)
    plt.xlabel("Depth")
    plt.ylabel("Wining Rate (%)")
    plt.title("Wining Rate between \n MiniMax Agent vs. Expecting Agent")
    plt.show()