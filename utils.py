import os
from tensorflow import keras
import numpy as np
import numba as nb
from tqdm import *
from utils import *
from random import *
import tensorflow as tf
import multiprocessing
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from my_layer import NR_GraphAttention

def load_triples(file_path,reverse = True):
    @nb.njit
    def reverse_triples(triples):
        reversed_triples = np.zeros_like(triples)
        for i in range(len(triples)):
            reversed_triples[i,0] = triples[i,2]
            reversed_triples[i,2] = triples[i,0]
            if reverse:
                reversed_triples[i,1] = triples[i,1] + rel_size
            else:
                reversed_triples[i,1] = triples[i,1]
        return reversed_triples
    
    with open(file_path + "triples_1") as f:
        triples1 = f.readlines()
        
    with open(file_path + "triples_2") as f:
        triples2 = f.readlines()
        
    triples = np.array([line.replace("\n","").split("\t") for line in triples1 + triples2]).astype(np.int64)
    node_size = max([np.max(triples[:,0]),np.max(triples[:,2])]) + 1
    rel_size = np.max(triples[:,1]) + 1
    
    return np.concatenate([triples,reverse_triples(triples)],axis=0), node_size, rel_size*2 if reverse else rel_size

def load_aligned_pair(file_path,ratio = 0.3):
    if "sup_ent_ids" not in os.listdir(file_path):
        with open(file_path + "ref_ent_ids") as f:
            aligned = f.readlines()
    else:
        with open(file_path + "ref_ent_ids") as f:
            ref = f.readlines()
        with open(file_path + "sup_ent_ids") as f:
            sup = f.readlines()
        aligned = ref + sup
    aligned = np.array([line.replace("\n","").split("\t") for line in aligned]).astype(np.int64)
    np.random.seed(12306)
    np.random.shuffle(aligned)
    return aligned[:int(len(aligned) * ratio)], aligned[int(len(aligned) * ratio):]

def generate_map(triples,node_size):
    triples = np.unique(triples,axis=0)
    
    selfs = np.array([[i,i] for i in range(node_size)])
    ent_ent = np.stack([triples[:,0],triples[:,2]],axis=1)
    ent_ent = np.concatenate([selfs,ent_ent],axis = 0)
    ent_ent = np.unique(ent_ent,axis=0)
    
    ent_rel = np.stack([triples[:,0],triples[:,1]],axis=1)
    ent_rel = np.unique(ent_rel,axis=0)
    
    return triples,ent_ent,ent_rel

@nb.njit()
def select_path(node_list,adj_dic,rel_weights,max_depth = 1,info = 1):
    rel_size = len(rel_weights)
    selected_triples,deque = [],[]
    node_dict,node_reverse_dict = {},[]
    ent_ent,ent_rel = {},{}
    vis = np.zeros(max(adj_dic.keys())+1,np.int32)
    
    for start_node in node_list:
        vis[start_node] = 1
        deque.append(start_node)
        node_reverse_dict.append(start_node)
        node_dict[start_node] = len(node_reverse_dict) - 1
    
    head,tail = 0,len(deque)
    while head < tail:
        now = deque[head]
        if vis[now] == max_depth:
            break
        ent_ent[(node_dict[now],node_dict[now])] = 1
        head += 1
        
        prob = [rel_weights[r] for r,t in adj_dic[now]]
        prob = np.array(prob)
        prob = prob/np.sum(prob)
        cumulative_distribution = np.cumsum(prob)
        cumulative_distribution = cumulative_distribution/cumulative_distribution[-1]
        t = int(info/np.sum(prob*prob))+1
        uniform_samples = np.random.rand(500)
        index = np.searchsorted(cumulative_distribution, uniform_samples, side="right")
        
        temp_vis = {}
        for i in index:
            rel,nei = adj_dic[now][i]
            if i in temp_vis:
                continue
            if len(temp_vis)>t:
                break
            temp_vis[i] = 1
            if vis[nei] == 0:
                tail += 1
                deque.append(nei)
                vis[nei] = vis[now] + 1
                node_reverse_dict.append(nei)
                node_dict[nei] = len(node_reverse_dict) - 1
            if vis[now] < max_depth-1:
                selected_triples.append((node_dict[now],rel,node_dict[nei]))
                selected_triples.append((node_dict[nei],rel+rel_size//2 if rel < rel_size//2 else rel-rel_size//2,node_dict[now]))
            ent_ent[(node_dict[now],node_dict[nei])] = 1
            ent_rel[(node_dict[now],rel)] = 1
            
    return np.array(selected_triples),list(ent_ent.keys()),list(ent_rel.keys()),node_dict,np.array(node_reverse_dict)


def CSLS_cal(Lvec,Rvec,evaluate,batch_size=1024):
    lsims,rsims = [],[]
    lavgs,ravgs = [],[]
    for epoch in range(len(Lvec) // batch_size + 1):
        lsim = tf.matmul(Lvec[batch_size*epoch:batch_size*(epoch+1)],Rvec.T)
        lavg = tf.reduce_mean(tf.nn.top_k(lsim,k=10)[0],axis=-1)
        lsims.append(np.array(lsim));lavgs.append(lavg);
        
        rsim = tf.matmul(Rvec[batch_size*epoch:batch_size*(epoch+1)],Lvec.T)
        ravg = tf.reduce_mean(tf.nn.top_k(rsim,k=10)[0],axis=-1)
        rsims.append(np.array(rsim));ravgs.append(ravg)
    results = []    

    for epoch in range(len(Lvec) // batch_size + 1):
        sim = rsims[epoch]
        sim = 2*sim - tf.transpose(tf.expand_dims(tf.concat(lavgs,axis=0),axis=1))
        sim = sim - tf.expand_dims(ravgs[epoch],axis=1)
        if not evaluate:
            #results.append(tf.argmax(sim,axis=-1))
            results.append(sim)
        else:
            rank = tf.argsort(-sim,axis=-1)
            ans_rank = np.array([i for i in range(epoch * batch_size,min((epoch+1) * batch_size,len(Lvec)))])
            results.append(tf.where(tf.equal(K.cast(rank,"int64"),tf.tile(np.expand_dims(ans_rank,axis=1),[1,len(Lvec)])))) 
    return np.array(np.concatenate(results,axis=0))
    
def GPU_test(Lvec,Rvec,batch_size=1024):
    results  = CSLS_cal(Lvec,Rvec,True,batch_size=batch_size)
    @nb.jit(nopython = True)
    def cal(results):
        hits1,hits5,hits10,mrr = 0,0,0,0
        for x in results[:,1]:
            if x < 1:
                hits1 += 1
            if x < 5:
                hits5 += 1
            if x < 10:
                hits10 += 1
            mrr += 1/(x + 1)
        return hits1,hits5,hits10,mrr
    hits1,hits5,hits10,mrr = cal(results)
    print([hits1/len(results),hits5/len(results),hits10/len(results),mrr/len(results)])
    return results