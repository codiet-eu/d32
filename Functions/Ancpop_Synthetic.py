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

class Ancpop_Simulation(object):
    '''
    A class for simulating random (causal) DAG based on synthetic datasets, where any DAG generator
    self.method would return the weighed/binary adjacency matrix of a DAG.

    Parameters
    ----------
    method: str, (linear or nonlinear), default='linear'
        Distribution for standard trainning dataset.
    File_PATH: str
        Save file path
    sem_type: str
        gauss, exp, gumbel, uniform, logistic (linear);
        mlp, mim, gp, gp-add, quadratic (nonlinear).
    nodes: series
        Notes of samples for standard trainning dataset
    edges: series
        Edges of samples for standard trainning dataset
    start: int
        Start number of samples for standard trainning dataset
    stop: int
        stop number of samples for standard trainning dataset
    step: int
        step number of samples for standard trainning dataset

    '''

    def __init__(self, method, File_PATH, sem_type, nodes, edges, start, stop, step):
        self.method = method
        self.File_PATH = File_PATH
        self.sem_type = sem_type
        self.nodes = nodes
        self.edges = edges
        self.start = start
        self.stop = stop
        self.step = step
        self.nodes_num = len(self.nodes)
        self.edges_num = len(self.edges)
        self.printname = self.method.capitalize()+' SEM Samples with ' + self.sem_type.capitalize() +' Noise'
        self.filename = self.method.capitalize()+'SEM_' + self.sem_type.capitalize() +'Noise'
        self.File_PATH_Base = self.File_PATH +'Result_'+ self.filename +'/'
        self.datasize = range(self.start, self.stop, self.step)
        self.datasize_num = len(self.datasize)

    def Ancpop_simulation_Test(self):
        ############################################## Create And Download Dataset ###################################
        self.File_PATH_Details = self.File_PATH_Base + 'Results_Details/'
        if not os.path.exists(self.File_PATH_Details):
            os.makedirs(self.File_PATH_Details)
        print('INFO: Created Result_'+self.method.capitalize()+' File!')
        tqdm_csv=os.listdir(self.File_PATH_Details)
        if len(tqdm_csv) != self.nodes_num*self.edges_num* self.datasize_num:
            print('INFO: Simulating '+self.printname +'!')
            self.Ancpop_simulation_estimate(self)
        else:
            print('INFO: Finished '+ self.printname+' simulation!')

        ################################################# Summary Dataset ############################################
        ################################## Insert/Download Dataset And Create Summary Table############################
        self.File_PATH_Heatmaps = self.File_PATH + 'Results_'+self.filename+ '/'
        if not os.path.exists(self.File_PATH_Heatmaps):
            os.makedirs(self.File_PATH_Heatmaps )
        self.Table_PATH_Summary = self.File_PATH_Heatmaps + 'Summary_'+self.filename+'.csv'
        if not os.path.exists(self.Table_PATH_Summary):
            print('INFO: Summarizing samples from '+'!')
            self.Summary_Results(self)
            # Table_Summary = self.Summary_Results()

        else:
            print('INFO: Finished '+ self.printname+' summary!')
            # Table_Summary = pd.read_csv(Table_PATH_Summary,header=0,index_col=0)

        ######################################### Create And Combine Dataset Summary #################################
        self.File_PATH_Results = self.File_PATH_Base + 'Results/'
        if not os.path.exists(self.File_PATH_Results):
            os.makedirs(self.File_PATH_Results)
        tqdm_results=os.listdir(self.File_PATH_Results)
        if len(tqdm_results) != (self.nodes_num+self.edges_num)*2:
            print('INFO: Preparing plot!')
            self.Type_Results(self, 'Nodes')
            self.Type_Results(self, 'Edges')

        ################################################  Create And Plot #############################################
        self.Plots_Type_ANCPOP(self,'Nodes')
        print('INFO: Finished plotting '+ self.printname + ' on nodes!')

        self.Plots_Type_ANCPOP(self,'Edges')
        print('INFO: Finished plotting '+ self.printname + ' on edges!')

        self.Plots_ANCPOP(self)
        print('INFO: Finished plotting '+ self.printname + '!')


    @staticmethod
    def Ancpop_simulation_estimate(self):
        duration_anm_ncpop = []
        f1_anm_ncpop = []
        df = pd.DataFrame(columns=['fdr', 'tpr', 'fpr', 'shd', 'nnz', 'precision', 'recall', 'F1', 'gscore'])
        for nn in range(6,14,3):
            for ne in range(10,21,5):
                weighted_random_dag = DAG.erdos_renyi(n_nodes=nn, n_edges=ne, seed=1)
                for ds in range(5,40,5):
                    sname = self.method+ '_gauss_'+str(nn)+'nodes_'+str(ne)+'edges_'+str(ds)+'DataSize'
                    path_check = os.path.exists(self.File_PATH_Base + 'Results_Details/' +sname+'.csv')
                    if not path_check:
                        dataset = IIDSimulation(W=weighted_random_dag, n=ds, method=self.method, sem_type=self.sem_type)
                        true_dag, data = dataset.B, dataset.X

                        t_start = time.time()
                        anmNCPO = ANM_NCPO(alpha=0.05)
                        anmNCPO.learn(data=data,causalmodelling='hidden_state1')

                        # plot predict_dag and true_dag
                        self.File_PATH_MetricsDAG = self.File_PATH_Base +'MetricsDAG/'
                        if not os.path.exists(self.File_PATH_MetricsDAG):
                          os.makedirs(self.File_PATH_MetricsDAG)
                        GraphDAG(anmNCPO.causal_matrix, true_dag, show=False, save_name = self.File_PATH_MetricsDAG +'Result_' +sname)
                        met = MetricsDAG(anmNCPO.causal_matrix, true_dag)
                        duration_anm_ncpop.append(time.time()-t_start)
                        f1_anm_ncpop.append(met.metrics['F1'])
                        print(sname+ 'is done!'+'F1 Score is'+ str(met.metrics['F1'])+'.' 'Time Duration is'+ str(time.time()-t_start))
                        #mm = pd.DataFrame(pd.DataFrame([mm]).append([met.metrics]).dropna(axis = 0, how ='any'))#.drop_duplicates(inplace= True)
                        df = pd.concat([df, pd.DataFrame([met.metrics])])
                        df.to_csv(self.File_PATH_Details +sname+'.csv', index=False)
                df1=df.assign(DataSize = ds)
                df1.to_csv(self.File_PATH_Base + 'summary_' + self.method+ '_gauss_'+str(nn)+'nodes_'+str(ne)+'edges' +'.csv', index=False)
        f1_anm_ncpop = np.array(f1_anm_ncpop)
        np.savetxt(self.File_PATH_Base + 'Summary_F1_' +self.filename+'.csv', f1_anm_ncpop, delimiter=',')
        #f1_anm_ncpop.to_csv(self.File_PATH_Base + 'Summary_F1_' + self.method +'_Gauss.csv', index=False)
        np.savetxt(self.File_PATH_Base + 'Duration_' +self.filename+'.csv', duration_anm_ncpop, delimiter=',')
        #duration_anm_ncpop.to_csv(self.File_PATH_Base + 'Duration_' + self.method + '_Gauss.csv', index=False)
        #f1_if_lds_mean = np.mean(f1_if_lds, axis=1)
        #f1_if_lds_std = np.std(f1_if_lds, axis=1)
        print(df1,f1_anm_ncpop, duration_anm_ncpop)

    @staticmethod
    def Summary_Results(self):
        f1_anm_ncpop = pd.DataFrame()
        tqdm=os.listdir(self.File_PATH_Details)
        for i in range(0,len(tqdm)):
            File_PATH = os.path.join(self.File_PATH_Details,tqdm[i])
            #entries = re.split("_", re.split("\.", tqdm[i])[0])
            self.method_nn = re.split("_",re.split("nodes_", re.split("\.", tqdm[i])[0])[0])
            ne = re.split("edges_",re.split("nodes_", re.split("\.", tqdm[i])[0])[1])[0]
            ds = re.split("D",re.split("edges_",re.split("nodes_", re.split("\.", tqdm[i])[0])[1])[1])[0]
            df = pd.read_csv(File_PATH)
            f1_anm_ncpop_nan = df.loc[:,'F1']
            if len([f1_anm_ncpop_nan == 0]) == len(f1_anm_ncpop_nan):
              f1_anm_ncpop_mean = 0.2
            else:
              f1_anm_ncpop_mean = round(np.nanmean(f1_anm_ncpop_nan), 3)
            f1_anm_ncpop = pd.concat((f1_anm_ncpop, pd.DataFrame(self.method_nn+[ne, ds, f1_anm_ncpop_mean])), axis=1)
        f1_result = pd.DataFrame(np.array(f1_anm_ncpop.T), columns=['Linear','Gauss','Nodes','Edges','DataSize','F1_Score'])
        f1_result.to_csv(self.Table_PATH_Summary,index=False)
        return f1_result

    @staticmethod
    def Type_Results(self, type):
        if type == 'Nodes':
            pivot = 'Edges'
        else:
            pivot = 'Nodes'
        f1_result =pd.read_csv(self.Table_PATH_Summary, header=0, index_col=0)
        group_obj = f1_result.groupby(type)#.agg('mean')
        for i in group_obj:
            print(i[0])
            f1_anm_ncpop_ = i[1].pivot(index=pivot,columns='DataSize',values='F1_Score')
            f1_anm_ncpop_result = f1_anm_ncpop_.reset_index()
            #print(f1_anm_ncpop_result)
            self.Table_PATH_Results = self.File_PATH_Results + 'Summary_'+type+'_'+str(i[0])
            f1_anm_ncpop_result.to_csv(self.Table_PATH_Results+".csv",index=False)
            np.save(self.Table_PATH_Results+".npy", f1_anm_ncpop_result)

    @staticmethod
    def Plots_Type_ANCPOP(self, type):
        if type == 'Nodes':
            type_num = self.nodes_num
            size = self.nodes
            labels = self.edges
            yaxes = 'Number of Edges'
        else:
            type_num = self.edges_num
            size = self.edges
            labels = self.nodes
            yaxes = 'Number of Variables'
        z = [[] for i in range(type_num)]
        fig, axes = plt.subplots(nrows=1, ncols=type_num,figsize=(20,4))
        for i in range(type_num):
            read_path = self.File_PATH_Base + 'Results/Summary_'+ type +'_'+str(size[i]) +'.csv'
            zz = pd.read_csv(read_path,header=0,index_col=0)
            z[i] = zz.to_numpy()
        z_min=np.min(z)
        z_max=np.max(z)
        for i in range(type_num):
            c = axes[i].imshow(z[i], cmap =plt.cm.bone_r, vmin = z_min, vmax = z_max, interpolation ='nearest', origin ='upper')
            axes[i].set_xlabel('length of time windows '+r'$T$',fontsize=10)
            axes[i].set_ylabel(yaxes,fontsize=10)
            axes[i].title.set_text('F1 score of '+ type + str(size[i]))
            positions = range(8)
            axes[i].yaxis.set_major_locator(ticker.FixedLocator(positions))
            axes[i].yaxis.set_major_formatter(ticker.FixedFormatter(labels))
            for ytick in axes[i].get_yticklabels():
                ytick.set_fontsize(4)
                #ytick.set_rotation(45)
            xlabels = self.datasize
            axes[i].xaxis.set_major_locator(ticker.FixedLocator(positions))
            axes[i].xaxis.set_major_formatter(ticker.FixedFormatter(xlabels))
            for xtick in axes[i].get_xticklabels():
                xtick.set_fontsize(4)
                #xtick.set_rotation(45)

        fig.colorbar(c, ax=axes.ravel().tolist())
        plt.savefig(self.File_PATH_Heatmaps +'Heatmap_ '+self.filename+'_'+type+'.pdf', bbox_inches='tight')

    @staticmethod
    def Plots_ANCPOP(self):
        max_num = np.max([self.nodes_num, self.edges_num])
        z = [[[] for i in range(max_num)]for j in range(2)]
        # z[1][2] [[[],[],[]],[[],[],[]]]
        fig, axes = plt.subplots(nrows=2, ncols=len(z[0]),figsize=(20,8))
        for j in range(2):
            if j == 0:
                type = 'Nodes'
                size = self.nodes
                labels = self.edges
                yaxes = 'Number of Edges'
            else:
                type = 'Edges'
                size = self.edges
                labels = self.nodes
                yaxes = 'Number of Variables'
            for i in range(max_num):
                read_path = self.File_PATH_Base + 'Results/Summary_'+ type +'_'+str(size[i]) +'.csv'
                zz = pd.read_csv(read_path,header=0,index_col=0)
                z[j][i] = zz.to_numpy()
            z_min=np.min(z[j])
            z_max=np.max(z[j])
            for i in range(max_num):
                c = axes[j][i].imshow(z[j][i], cmap =plt.cm.bone_r, vmin = z_min, vmax = z_max, interpolation ='nearest', origin ='upper')
                axes[j][i].set_xlabel('length of time windows '+r'$T$',fontsize=10)
                axes[j][i].set_ylabel(yaxes,fontsize=10)
                axes[j][i].title.set_text('F1 score of '+ type + str(size[i]))
                positions = range(8)
                axes[j][i].yaxis.set_major_locator(ticker.FixedLocator(positions))
                axes[j][i].yaxis.set_major_formatter(ticker.FixedFormatter(labels))
                for ytick in axes[j][i].get_yticklabels():
                    ytick.set_fontsize(4)
                    #ytick.set_rotation(45)
                xlabels = self.datasize
                axes[j][i].xaxis.set_major_locator(ticker.FixedLocator(positions))
                axes[j][i].xaxis.set_major_formatter(ticker.FixedFormatter(xlabels))
                for xtick in axes[j][i].get_xticklabels():
                    xtick.set_fontsize(4)
                    #xtick.set_rotation(45)
        title = 'Performance of '+ self.printname
        fig.colorbar(c, ax=axes.ravel().tolist())
        fig.suptitle(title, fontsize=16)
        #plt.title(title, fontdict=None, loc='center', pad=None)
        plt.savefig(self.File_PATH_Heatmaps +'Heatmap_ '+self.filename+'.pdf', bbox_inches='tight')

if __name__ == "__main__":
    ###################################################################################################################
    ########################### SETTING Method, SEM Noise_TYPE, File_PATH, nodes, edges, and datasize ##########################
    method = 'nonlinear'
    File_PATH = '/content/drive/MyDrive/Colab Notebooks/Causality_NotesTest/'
    sem_type = 'gp-add'
    nodes = range(6,14,3)
    edges = range(10,21,5)
    start = 5
    stop = 40
    step = 5
    st = Ancpop_Simulation(method, File_PATH, sem_type, nodes, edges, start, stop, step)
    st.Ancpop_simulation_Test()
