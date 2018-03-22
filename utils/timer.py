from __future__ import division
import time
import matplotlib.pyplot as plt


class Timer(object):

    """Timer tool recording running time
   
    Examples:
        >>> from utils import Timer
        >>> timer = Timer()
        >>> timer.start()
        >>> for i in range(10):
        >>>     # do something
        >>>     ...
        >>>     timer.tictoc('p1')
        >>>     # do something
        >>>     ...
        >>>     timer.tictoc('p2')
        >>>     # do something
        >>>     ...
        >>>     timer.tictoc('p3')
        >>> timer.end()
        >>> timer.log(show=True)
    
    """

    def __init__(self):
        self.startpoint = 0
        self.endpoint = 0
        self.ticpoint = 0
        self.tic_dict = {}
        self.steps = 0


    def start(self):
        print('===> tic...toc...')
        self.startpoint = time.time()
        self.ticpoint = time.time()


    def tictoc(self, name):
        tic = time.time() - self.ticpoint
        if name in self.tic_dict.keys():
            self.tic_dict[name] += tic
        else:
            self.tic_dict[name] = tic
        self.ticpoint = time.time()
        self.steps += 1


    def end(self):
        self.endpoint = time.time()
        self.steps = int(self.steps / len(self.tic_dict.keys()))


    def ms(self, input):
        return int(round(input*1000))


    def log(self, show=True, average=True):
        time_sum = self.endpoint - self.startpoint
        print('Running for {} loops'.format(self.steps))
        print('Time sum: %d ms' %  self.ms(time_sum))
        print('Time ave: %d ms' %  self.ms(time_sum/self.steps))

        sorted_tics = sorted(self.tic_dict.items(), \
                lambda x, y: cmp(x[1], y[1]), reverse=True)
        for name, time in sorted_tics:
            if average:
                print('[ave] {}: {}ms, {}%'.format(name, self.ms(time/self.steps), int(time*100/time_sum)))
            else:
                print('[sum] {}: {}ms, {}%'.format(name, self.ms(time), int(time*100/time_sum)))

        if show:
            self.show()


    def show(self):
        sorted_tics = sorted(self.tic_dict.items(), \
                lambda x, y: cmp(x[1], y[1]), reverse=True)
        labels = []
        times = []
        for label, time in sorted_tics:
            labels.append(label + ' {}ms'.format(self.ms(time/self.steps)))
            times.append(time)

        max_ind = times.index(max(times))
        explode = [0 for x in range(len(times))]
        explode[max_ind] = 0.1

        fig, ax = plt.subplots()
        ret = ax.pie(times, explode=tuple(explode), autopct='%1.1f%%',
                                shadow=True, startangle=90)
        ax.legend(ret[0], labels, loc="best")

        plt.tight_layout()
        plt.axis('equal')  
        plt.show()
