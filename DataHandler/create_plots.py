import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import norm
import seaborn as sns
from scipy import stats
from scipy import optimize
import pylab as py

##########################################################################
"""
    IMPORT DATA
"""

def create_all_hist(Model, dataset, TXT_PATH):

    PATH = '/home/exjobb/results/' + Model + dataset + '/distrubation_readable.cvs'

    df = pd.read_csv(PATH)

    # The path to the plots and textfile
    PATH = '/home/exjobb/results/' + Model + dataset

    # Open the textfile
    # f = open(PATH + '/properties.txt', "w+")
    f = open(TXT_PATH, "a")
    # f.write("DISTRIBUTION PROPERTIES \r\n")
    f.write("##################################################### \r\n\r\n")
    f.write("Classes IoU then mIoU\r\n")
    f.write("mean, std, min,  max\r\n")

    for metric in df.keys()[1:]:
        acc_vec = df[str(metric)].values * 100

        # Call create_hist functions
        create_hist(acc_vec[~np.isnan(acc_vec)], str(metric), PATH + '/' + str(metric) + '_', f, Model, dataset)

    f.close()

##########################################################################


def create_hist(data, titel_text, PATH, f, model, dataset):
    if len(data) == 0:
        print(titel_text + ' has zero length')
    else:
        mu = np.nanmean(data)    # mean of distribution
        sigma = np.nanstd(data)  # standard deviation of distribution
        mini = np.nanmin(data)    # min of distribution
        maxi = np.nanmax(data)    # max of distribution

        nr_bins = 20

        bin = [(100//nr_bins)*i for i in range(nr_bins + 1)]

        fig, ax = plt.subplots()

        # manipulate
        vals = ax.get_xticks()
        mean_val = round(mu, 1)

        ax.set_xticklabels(['{:.0f}%   '.format(x*100) for x in vals])

        # the histogram of the data
        ax.hist(data, bins=bin, rwidth=0.8, color='#B7C1E5')

        ax.tick_params(labelsize=14)

        ax.set_xlabel('Accuracy', fontsize=18)
        ax.set_ylabel('Number of samples', fontsize=18)

        # Tweak spacing to prevent clipping of ylabel
        fig.tight_layout()

        plt.grid(True)

        plt.xlim(0, 100)

        plt.axvline(x=mu, color='k', linestyle='--')

        # these are matplotlib.patch.Patch properties
        props = dict(facecolor='white', alpha=1)

        textstr = '\n'.join((
            r'$\mu=%.1f$%%' % (mean_val, ),
            r'$\sigma=%.1f$%%' % (sigma, )))

        # place a text box in upper left in axes coords
        ax.text(0.79, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)

        # Save figure
        plt.savefig(PATH + model + dataset + '.png')

        # Update textfile
        f.write("{:5} {:5} {:5} {:5} \r\n".format(str(round(mu, 1)), str(round(sigma, 1)), str(round(mini, 1)), str(round(maxi, 1))))


def loss_graph(data, Model, dataset):
    PATH = '/home/exjobb/results/' + Model + dataset

    # var = round(np.max(data) * 1.3)
    #
    # data.insert(0, 10000)



    plt.plot(data, color='#607c8e')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    # plt.ylim(0, var)
    # plt.xlim(0, len(data)-1)

    # Save figure
    plt.savefig(PATH + '/val_loss_over_epochs.png')


# plt.show()

