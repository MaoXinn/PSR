from __future__ import absolute_import

from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dropout, LeakyReLU
import tensorflow as tf
import numpy as np

class TokenEmbedding(tf.keras.layers.Embedding):
    """Embedding layer with weights returned."""

    def compute_output_shape(self, input_shape):
        return self.input_dim, self.output_dim

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs):
        return self.embeddings

class NR_GraphAttention(Layer):

    def __init__(self,
                 node_size,
                 rel_size,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 activation=None,
                 use_bias=False,
                 kernel_initializer='uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')
        self.node_size = node_size
        self.rel_size = rel_size
        self.attn_heads = attn_heads  
        self.attn_heads_reduction = attn_heads_reduction  
        self.activation = activations.get(activation) 
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        self.biases = []        
        self.attn_kernels = []  

        super(NR_GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        ent_hidden_size = input_shape[1][-1]
        rel_hidden_size = input_shape[2][-1]
            
        for head in range(self.attn_heads):                
            attn_kernel = self.add_weight(shape=(1*rel_hidden_size ,1),
                                   initializer=self.attn_kernel_initializer,
                                   regularizer=self.attn_kernel_regularizer,
                                   constraint=self.attn_kernel_constraint,
                                   name='attn_kernel_{}'.format(head))
            self.attn_kernels.append(attn_kernel)
            
            if self.use_bias:
                self.bias = self.add_weight(shape=(1,ent_hidden_size),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint,
                                       name='bias_{}'.format(head))
                self.biases.append(bias)
                
        self.built = True
        
    
    def call(self, inputs):
        triples,features,rel_emb = inputs
        triples = K.cast(K.squeeze(triples,0),dtype="int32")
                
        features_list = []
        for head in range(self.attn_heads):
            att_kernel = self.attn_kernels[head]  
            sub_embs = K.gather(indices=triples[:,0],reference=features)
            edge_embs = K.gather(indices=triples[:,1],reference=rel_emb)
            obj_embs = K.gather(indices=triples[:,2],reference=features)
            
            bias = tf.reduce_sum(obj_embs * tf.nn.l2_normalize(edge_embs,1), 1, keepdims=True) * edge_embs
            obj_embs = obj_embs - 2 * bias
            
            att_value = K.exp(K.gather(indices=triples[:,1],reference=K.dot(rel_emb,att_kernel)))
            new_embs = tf.math.segment_sum(att_value*obj_embs,triples[:,0])
            new_embs = new_embs / (tf.math.segment_sum(att_value,triples[:,0]))
            
            features_list.append(new_embs)

        if self.attn_heads_reduction == 'concat':
            features = K.concatenate(features_list)  # (N x KF')
        else:
            features = K.mean(K.stack(features_list), axis=0)

        features = self.activation(features)
        return features
    
    def compute_output_shape(self, input_shape):    
        node_shape = self.node_size, input_shape[1][-1]
        return node_shape