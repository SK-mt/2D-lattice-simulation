#!/usr/bin/env python
# coding: utf-8

# In[8]:


from enum import Enum
import random
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from scipy.special import comb
from math import factorial


# In[11]:


class State(Enum):
    S0 = 'State0'
    S2 = 'State2'
    S3 = 'State3'

class DiffusionModel:
    def __init__(self, N):
        self.N = N**2
        # ネットワーク設定
        self.network = nx.grid_2d_graph(N,N)
        
        self.qi_d = {}
        for i in self.network.nodes():
            qi = 1
#            gamma = np.random.rand ()
            self.qi_d[i] = qi
        self.target = []
        
    def update_SIR_nums(self, s, i, r):
        self.S_list.append(s)
        self.I_list.append(i)
        self.R_list.append(r)
        
    def init_source(self):
        node_list = list(self.network.nodes)
        # 初期状態は全員"S"
        for i in node_list:
            self.network.nodes[i]['state'] = State.S0
        
        self.network.nodes[(0,0)]['state'] = State.S2
        self.received_nodes.append(i)
        self.event_list[i] = 0
        
        self.update_SIR_nums(self.N, 1, 0)
        
    def spread(self, Lambda, T=50000):
        # イベントリスト
        self.event_list = {}
        # 時刻tでの各状態のエージェントリスト
        self.unreceived_nodes = []
        self.received_nodes = []
        self.broadcasted_nodes = []
        # ある状態のエージェント数のリスト
        self.S_list = []
        self.I_list = []
        self.R_list = []
        # 時刻リスト
        self.Time = []
        # グラフの色
        self.colors = {"State2": 'black'}
        # 初期感染者(x)
        self.init_source()
        
        time = 0
        self.Time.append(time)
        while len(self.event_list) > 0:
            
            if time > T:
                break
            
            # 最新の各ノード数を取得
            new_S_num = self.S_list[-1]
            new_I_num = self.I_list[-1]
            new_R_num = self.R_list[-1]
            
            # イベント決定
            event_node, event_time = min(self.event_list.items(), key=lambda x: x[1])
            node = event_node
            self.Time.append(event_time)
            
            for neighbor in self.network.neighbors(node):
                qi = self.qi_d[neighbor]
                if self.network.nodes[neighbor]['state'] == State.S0 and np.random.random() < qi:
                        self.network.nodes[neighbor]['state'] = State.S2
                        if neighbor == (30, 10):
                            self.target.append(time)
                        self.event_list[neighbor] = time + (-np.log(1-np.random.rand())*(1/Lambda))
                        new_I_num += 1
                        new_S_num -= 1
                        self.received_nodes.append(neighbor)
                
            self.broadcasted_nodes.append(node)
            self.received_nodes.remove(node)
            self.network.nodes[node]['state'] = State.S3
            self.event_list.pop(node)
            new_R_num += 1
            new_I_num -= 1
            
            self.update_SIR_nums(new_S_num, new_I_num, new_R_num)
            time = event_time
        
    def pltfig(self):
        #グラフ描画
        # figure()でグラフを描画する領域を確保，figというオブジェクトにする．
        fig = plt.figure(figsize=(5,5))
        # add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
        ax1 = fig.add_subplot(1, 1, 1) 
        ax2 = ax1.twinx()   # x軸を共有
        # 1つ目のグラフを描画
        ax1.set_xlabel('t') #x軸ラベル
        ax1.plot(self.Time, self.R_list, self.colors[State.S2.value], label=State.S3.value)
        h1, l1 = ax1.get_legend_handles_labels()
        # 2つ目のグラフを描画
        ax2.plot(self.Time, self.I_list, self.colors[State.S2.value], linestyle = "dashed", label=State.S2.value)
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='upper right') # ax1とax2の凡例のhandlerとlabelのリストを結合
        plt.show()
        
    def AU(self, Lambda, n, m, k, t):
        alpha = []
        u = []
        for i in range(1, k+1):
            if i == k:
                alpha_i = Lambda**i
                alpha.append(alpha_i)
            else:
                alpha_i = (Lambda**i) * ((-1)**(k-i)) * math.prod((n+m)-(q+1) for q in range(i, k)) / math.factorial(k-i)
                alpha.append(alpha_i)
            u_i = (t**(i-1)) / math.factorial(i-1)
            u.append(u_i)
        A = np.array(alpha)
        U = np.array(u).T
        return A, U

    def BV(self, Lambda, n, m, k, t):
        beta = []
        v = []
        for j in range(1, n+m-k+1):
            if j == n+m-k:
                beta_j = ((-1)**k) * (Lambda**j)
                beta.append(beta_j)
            else:
                beta_j = (Lambda**j) * ((-1)**k) * math.prod((n+m)-(r+1) for r in range(j, n+m-k)) / math.factorial(n+m-k-j)
                beta.append(beta_j)
            v_j = (t**(j-1)) / math.factorial(j-1)
            v.append(v_j)
        B = np.array(beta)
        V = np.array(v).T
        return B, V

    def dpdt(self, Lambda, n, m, t):
        n_list = []
        for k in range(1, n+1):
            A, U = self.AU(Lambda, n, m, k, t)
            B, V = self.BV(Lambda, n, m, k, t)
            binom1 = comb(n+m-k-1, m-1, exact=True)
            sum_n = binom1 * ( A@U * np.exp(-Lambda*t) + B@V * np.exp(-2*Lambda*t) )
            n_list.append(sum_n)
        m_list = []
        for k in range(1, m+1):
            A, U = self.AU(Lambda, n, m, k, t)
            B, V = self.BV(Lambda, n, m, k, t)
            binom2 = comb(n+m-k-1, n-1, exact=True)
            sum_m = binom2 * ( A@U * np.exp(-Lambda*t) + B@V * np.exp(-2*Lambda*t) )
            m_list.append(sum_m)
        dpdt = sum(n_list) + sum(m_list)
        return dpdt

    def numerical_integration(self, Lambda, n, m, t):
        N = 10000
        t_0 = 0
        t_n = t
        h = (t_n - t_0) / N
        p = (h/2) * (self.dpdt(Lambda, n, m, t_0) + 2*sum(self.dpdt(Lambda, n, m, h*i) for i in range(1,N-1)) + self.dpdt(Lambda, n, m, t_n))
        return p

    def pt(self, Lambda, n_or_m, t):
        p = 1 - sum(((Lambda*t)**k) * np.exp(-Lambda*t) / math.factorial(k) for k in range(0, n_or_m))
        return p
        
    def target_object(self):
        Lambda = 1
        p_list = []
        t_list = []
        t = 0
        while t < 40:
            t_list.append(t)
            p = self.numerical_integration(Lambda, 30, 10, t)
            p_list.append(p)
            t += 1
            #t += (-np.log(1-np.random.rand())*(1/Lambda))
        
        #plt.figure(figsize=(5,5))
        plt.plot(t_list, p_list, color='blue')
        sns.ecdfplot(data=self.target, color='black')
        #plt.hist(self.target, color='black', stacked=True, histtype="step", cumulative=True, density=True)
        plt.xlim(0, 40)
        plt.xlabel("t")
        plt.title("(30, 10)")
        plt.show()
    
if __name__ == '__main__':
    max_season = 100
    N = 50
    Lambda = 1
    
    sir = DiffusionModel(N)
    for season in range(1, max_season+1):
        season += 1
        sir.spread(Lambda)
        #print("season",season)
        #sir.pltfig()
    sir.target_object()


# In[ ]:




