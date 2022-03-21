"""
Application : Kaggle Assignment 
File name   : analysis.py
Authors     : Jacob Summerville
Description : This file performs a linear regression and correlation matrix 
              analysis with the provided dataset
"""

import datetime
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
from scipy import stats
import statsmodels.api as sm
import seaborn as sn
import matplotlib.pyplot as plt

# Constants for analysis
PVALUE_CUTOFF = 0.05
CORRELATION_CUTOFF = 0.5
HIGH_CORRELATION_CUTOFF = 0.7


def sort_pvalue(val):
    """ Sort p-values by value """
    return val[1] 


class Analysis:
    retained_metrics = {
        'pvalue': {},
        'correlation': {}
    }

    def __init__(self, filename, sheet, verbose = None):
        self.raw_data_file = filename
        self.sheet_name = sheet
        self.verbose = verbose

        self.create_output_file()
        self.read_file_data()
        self.linear_regression()
        self.pvalues()
        self.correlated()
        self.correlation_matrix()
        self.correlation_chart()
        self.keep_similar()
        self.close_output_file()


    def create_output_file(self):
        """ Create the file to save output data to """
        if not os.path.isdir('output/'):
            os.mkdir('output/')

        # name the file using the current date and time
        today = datetime.datetime.now()
        self.current_time = today.strftime("%Y%m%d_%H%M%S")
        filename = 'output/' + self.current_time + '.txt'
        self.output_file = open(filename, 'w')


    def write_output(self, text):
        """ Write text to the output file """
        self.output_file.write(text + '\n')


    def close_output_file(self):
        """ Save and close output file """
        self.output_file.close()


    def read_file_data(self):
        """ Read the excel file data """
        df = pd.read_excel(self.raw_data_file, sheet_name=self.sheet_name)
        self.data_columns = df.columns

        # half of the dataset should be used to train the model
        half_point = (len(df[self.data_columns[0]]) // 2)
        self.train_df = df.iloc[:half_point]


    def linear_regression(self):
        """ Perform linear regression and output the summary """
        X = self.train_df[self.data_columns[1:]]
        Y = self.train_df['SalePrice']

        X = sm.add_constant(X)
        self.regression_model = sm.OLS(Y, X).fit()

        if self.verbose:
            self.write_output(str(self.regression_model.summary()))
        else:
            self.write_output('R-Squared: ' + '{:3f}'.format(self.regression_model.rsquared))


        # generate the correlation matrix dataframe
        self.correlation_df = self.train_df.corr()


    def pvalues(self):
        """ Find p-values and display the ones above and below the cutoff value """
        pvalues_meeting_req = []

        # get all p-values below the specified cutoff
        for i in range(1, len(self.regression_model.pvalues)):
            if self.regression_model.pvalues[i] < PVALUE_CUTOFF:
                self.retained_metrics['pvalue'][self.data_columns[i]] = self.regression_model.pvalues[i]

        if self.verbose:
            self.write_output('\n' + ('=' * 40))

        self.write_output('\nKeep based on p-value: (Cutoff of ' + str(PVALUE_CUTOFF) + ')')
        self.write_output('---------------------------------------')

        if len(self.retained_metrics['pvalue']) == 0:
            self.write_output('None')
            return

        # sort by pvalue
        self.retained_metrics['pvalue'] = dict(sorted(self.retained_metrics['pvalue'].items(), 
                                                key=lambda item: item[1]))

        for metric, pvalue in self.retained_metrics['pvalue'].items():
            self.write_output('{:15s}: {:.2e}'.format(metric, pvalue))
            


    def correlation_matrix(self):
        """ Create a correlation matrix and display it """
        sn.heatmap(self.correlation_df, vmin=0.5, vmax=1, annot=True, cmap='BrBG')

        figure = plt.gcf()
        figure.set_size_inches(30, 20)
        plt.title("Correlation Matrix")
        plt.savefig('output/' + self.current_time + '_matrix.pdf')
        
        # clear plot to prevent future interference
        plt.clf()


    def correlation_chart(self):
        """ Create a correlation matrix with Sales Price """
        correlation_values = self.correlation_df[['SalePrice']].sort_values(by='SalePrice', ascending=False)

        heatmap = sn.heatmap(correlation_values, vmin=-1, vmax=1, annot=True, cmap='BrBG')
        heatmap.set_title('Features Correlating with Sale Price', pad=16);

        figure = plt.gcf()
        figure.set_size_inches(15, 12)
        plt.savefig('output/' + self.current_time + '_correlation.pdf')


    def correlated(self):
        """ Analyze correlated values and parse the ones to keep """
        sorted_df = self.correlation_df.sort_values(by='SalePrice', ascending=False)

        correlated_columns = sorted_df.index[sorted_df['SalePrice'] > CORRELATION_CUTOFF].tolist()
        self.retained_metrics['correlation'] = dict.fromkeys(correlated_columns , None)

        self.write_output('\nKeep based on correlation: (Cutoff of ' + str(CORRELATION_CUTOFF) + ')')
        self.write_output('------------------------------------------')

        if len(self.retained_metrics['correlation']) == 0:
            self.write_output('None')
            return

        for metric in correlated_columns:
            if metric == 'SalePrice':
                continue

            correlation_value = sorted_df['SalePrice'].loc[[metric]].item()
            formatted_output = '{:15s}: {:.2f}'.format(metric, correlation_value)

            # find multicollinearity between the primary metrics
            secondary_correlations = sorted_df.index[sorted_df[metric] > HIGH_CORRELATION_CUTOFF].tolist()
            mc_metrics = []

            for mc_metric in secondary_correlations: 
                if mc_metric in self.retained_metrics['correlation'] and mc_metric != 'SalePrice' and mc_metric != metric:
                    mc_metrics.append(mc_metric)

            if len(mc_metrics) > 0:
                self.write_output(formatted_output + ' (Multicollinearity: ' + ', '.join(mc_metrics) + ')')
            else:
                self.write_output(formatted_output)


    def keep_similar(self):
        """ List off p-value and correlation values that should both be kept """
        self.write_output('\nKeep both:')
        self.write_output('--------------')

        match_found = False

        for metric in self.retained_metrics['correlation']:
            if metric in self.retained_metrics['pvalue']:
                self.write_output(metric)
                match_found = True

        if not match_found:
            self.write_output('None')
    

if __name__ == '__main__':
    Analysis('train.xlsx', 'Train-Cleaned')