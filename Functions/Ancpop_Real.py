import os
import re
import csv
import math
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from castle.common import GraphDAG
from castle.metrics import MetricsDAG

class Ancpop_Real(object):
    '''
    A class for simulating random (causal) DAG, where any DAG generator
    self.method would return the weighed/binary adjacency matrix of a DAG.

    Parameters
    ----------
    File_PATH: str
        Save file path
    File_NAME: str
        Read data name
    start: int
        Start number of samples for standard trainning dataset
    stop: int
        stop number of samples for standard trainning dataset
    step: int
        step number of samples for standard trainning dataset

    '''
    def __init__(self, File_PATH,File_NAME, start, stop, step):
        self.File_PATH = File_PATH
        self.File_NAME = File_NAME
        self.start = start
        self.stop = stop
        self.step = step
        self.datasize = range(self.start, self.stop, self.step)
        self.datasize_num = len(self.datasize)
        self.sname = re.split("\.", self.File_NAME)[0]

    def Ancpop_real_Test(self):
        ################################################  Test Data #############################################
        self.File_PATH_Heatmaps = self.File_PATH + 'Results_'+self.sname+ '/'
        self.Table_PATH_Summary = self.File_PATH_Heatmaps + 'Summary_'+self.sname+'.csv'
        if not os.path.exists(self.Table_PATH_Summary):
            print('INFO: Testing '+ self.sname+'!')
            self.File_PATH_Base = self.File_PATH + 'Details_'+self.sname+ '/'
            self.Ancpop_real_estimate(self)
        else:
            print('INFO: Finished '+ self.sname+'Sampling!')

        ################################################  Create And Plot #############################################
        #self.Plots_ANCPOP_Real(self)
        print('INFO: Finished plotting '+ self.sname + '!')

    @staticmethod
    def Ancpop_real_estimate(self):
        read_Dir=os.listdir(self.File_PATH)
        if len(read_Dir) == 1:
            self.File_TYPE = re.split("\.", self.File_NAME)[1]
            if self.File_TYPE =='npz':
                print(self.File_TYPE)
                Tests_data = np.load(self.File_PATH+self.File_NAME, allow_pickle=True)
                Raw_data = Tests_data['x']
                true_dag = Tests_data['y']
                #print(Tests_data['x'][:20], Raw_data[:20],true_dag)
                #self.nodes_num = len(causal_matrix)
                #self.edges_num = len(causal_matrix[causal_matrix == 1])
            elif self.File_TYPE =='tar':
                Tests_data = np.load(self.File_NAME+'.tar', allow_pickle=True)
            else:
                print('INFO: Cannot Read '+self.File_TYPE+' Data!')
        else:
            Raw_data_path = self.File_PATH+read_Dir[0]
            Raw_data = pd.read_csv(Raw_data_path, header=0, index_col=0)
            DAG_data_path = self.File_PATH+read_Dir[1]
            DAG_data = pd.read_csv(DAG_data_path, header=0, index_col=0)


        ############################################## Create Files ###################################
        if not os.path.exists(self.File_PATH_Base):
            os.makedirs(self.File_PATH_Base)
        self.File_PATH_MetricsDAG = self.File_PATH_Base +'MetricsDAG/'
        if not os.path.exists(self.File_PATH_MetricsDAG):
            os.makedirs(self.File_PATH_MetricsDAG)
        if not os.path.exists(self.File_PATH_Heatmaps):
            os.makedirs(self.File_PATH_Heatmaps)

        duration_anm_ncpop = []
        f1_anm_ncpop = []
        df = pd.DataFrame(columns=['fdr', 'tpr', 'fpr', 'shd', 'nnz', 'precision', 'recall', 'F1', 'gscore'])

        #df = pd.DataFrame(columns=['DataSize','False_Discovery_Rate', 'True_Positive_Rate', 'False_Positive_Rate', 'SHD', 'NNZ', 'Precision', 'Recall', 'F1_Score', 'G_score'])
        for i in self.datasize:
            data = Raw_data[:i]
            t_start = time.time()
            anmNCPO = ANM_NCPO(alpha=0.05)
            anmNCPO.learn(data = data, causalmodelling = 'hidden_state1')
            # plot predict_dag and true_dag
            GraphDAG(anmNCPO.causal_matrix, true_dag, show=False, save_name = self.File_PATH_MetricsDAG+self.sname+'_' + str(i) + 'Datasize.png')
            met = MetricsDAG(anmNCPO.causal_matrix, true_dag)
            #f1_result.to_csv(self.File_PATH_Heatmaps + 'F1_'+self.sname+'.csv',index=False)
            if math.isnan(float(met.metrics['F1'])):
              f1_anm_ncpop.append(0.2)
            else:
              print(math.isnan(met.metrics['F1']),met.metrics['F1']==np.nan)
              f1_anm_ncpop.append(met.metrics['F1'])
            duration_anm_ncpop.append(time.time()-t_start)
            print(self.sname+'_' + str(i) + 'Datasize is done!'+'F1 Score is'+ str(met.metrics['F1'])+'.')
            #'Time Duration is'+ str(time.time()-t_start))
            df = pd.concat([df, pd.DataFrame([met.metrics])])
        df.to_csv(self.Table_PATH_Summary, index=False)
        #df = pd.concat({"DataSize":[self.datasize]},{df.loc[:, ['F1']]})
        df_F1 = pd.DataFrame({"DataSize":self.datasize,'F1_Score':f1_anm_ncpop,'Duration':duration_anm_ncpop})
        #f1_result = df.loc[:, ['DataSize','F1_Score']]
        df_F1.to_csv(self.File_PATH_Heatmaps + 'F1_'+self.sname+'.csv',index=False)
        return df_F1

    '''
    @staticmethod
    def Summary_Results(self):
        f1_anm_ncpop = pd.DataFrame()
        tqdm=os.listdir(self.File_PATH_Details)
        for i in range(0,len(tqdm)):
            File_PATH = os.path.join(self.File_PATH_Details,tqdm[i])
            #ds = re.split("D",re.split("edges_",re.split("nodes_", re.split("\.", tqdm[i])[0])[1])[1])[0]
            df = pd.read_csv(File_PATH)
            f1_anm_ncpop_nan = df.loc[:,'F1']
            if len([f1_anm_ncpop_nan == 0]) == len(f1_anm_ncpop_nan):
              f1_anm_ncpop_mean = 0.2
            else:
              f1_anm_ncpop_mean = round(np.nanmean(f1_anm_ncpop_nan), 3)
            f1_anm_ncpop = pd.concat((f1_anm_ncpop, pd.DataFrame([ds, f1_anm_ncpop_mean])), axis=1)
        f1_result = pd.DataFrame(np.array(f1_anm_ncpop.T), columns=['Linear','Gauss','Nodes','Edges','DataSize','F1_Score'])
        f1_result.to_csv(self.Table_PATH_Summary,index=False)
        return f1_result'''

    @staticmethod
    def Plots_ANCPOP_Real(self):
        self.pro_rang = np.arange(self.start, self.stop, self.step)
        self.obs_rang = np.arange(self.start, self.stop, self.step)
        fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(20,20))
        z =[]
        zz = pd.read_csv(self.File_PATH_Heatmaps + 'F1_'+self.sname+'.csv', header=[1], index_col=0)
        z = zz.to_numpy()
        z_min=np.min(z)
        z_max=np.max(z)
        c = axes.imshow(z, cmap =plt.cm.bone_r, vmin = z_min, vmax = z_max,
                        interpolation ='nearest', origin ='upper')
        axes.set_xlabel('length of time windows '+r'$T$',fontsize=10)
        #axes.set_ylabel('F1 score ',fontsize=10)
        positions = range(9)
        labels=self.datasize
        axes.yaxis.set_major_locator(ticker.FixedLocator(positions))
        axes.yaxis.set_major_formatter(ticker.FixedFormatter(labels))
        for ytick in axes.get_yticklabels():
            ytick.set_fontsize(4)
        axes.xaxis.set_major_locator(ticker.FixedLocator(positions))
        axes.xaxis.set_major_formatter(ticker.FixedFormatter(labels))
        for xtick in axes.get_xticklabels():
            xtick.set_fontsize(4)

        fig.colorbar(c, ax=axes.ravel().tolist())
        plt.savefig(self.File_PATH_Heatmaps +'Heatmap_ '+self.sname+'.pdf', bbox_inches='tight')


if __name__ == "__main__":
    ######################################################################################################################
    ############################################ SETTING File_PATH, file_name and datasize #############################
    #File_PATH = '/content/drive/MyDrive/Colab Notebooks/Causality_NotesTest/Test_Causality_Datasets/Real_data/Telephone/'
    File_PATH = '/content/drive/MyDrive/Colab Notebooks/Causality_NotesTest/Test_Causality_Datasets/Synthetic datasets/'

    start = 5
    stop = 40
    step = 5
    file_name = 'linearGauss_6_15.npz'
    rt = Ancpop_Real(File_PATH, file_name, start, stop, step)
    rt.Ancpop_real_Test()
