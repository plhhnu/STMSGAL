# import tensorflow.compat.v1 as tf
import tensorflow as tf2
tf = tf2.compat.v1
tf.disable_v2_behavior()
from sklearn.metrics import f1_score, accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn import cluster
import numpy as np

class GATE():

    def __init__(self, hidden_dims, alpha=0.5, nonlinear=True, weight_decay=0.0001,batchsize = 0,n_cluster = 6,
                 reg_ssc_coef=1.0,cost_ssc_coef=1.0,L_self_supervised_coef = 1.0):
        self.n_layers = len(hidden_dims) - 1
        self.hidden_dims = hidden_dims
        self.alpha = alpha
        self.W, self.v, self.prune_v = self.define_weights(hidden_dims)
        self.C = {}
        self.prune_C = {}
        self.nonlinear = nonlinear
        self.weight_decay = weight_decay
        self.batch_size = batchsize
        self.n_cluster = n_cluster
        self.reg_ssc_coef = reg_ssc_coef
        self.cost_ssc_coef = cost_ssc_coef
        self.L_self_supervised_coef = L_self_supervised_coef

    def __call__(self, A, prune_A, X):
        # multi-scale self-expression matrix
        Coef_0 = tf.Variable(1.0e-4 * tf.ones([self.batch_size, self.batch_size], tf.float32), name='Coef_0')
        Coef_1 = tf.Variable(1.0e-4 * tf.ones([self.batch_size, self.batch_size], tf.float32), name='Coef_1')
        Coef_2 = tf.Variable(1.0e-4 * tf.ones([self.batch_size, self.batch_size], tf.float32), name='Coef_2')

        weight_0 = tf.Variable(1.0, name='weight_0')
        weight_1 = tf.Variable(1.0, name='weight_1')
        weight_2 = tf.Variable(1.0, name='weight_2')

        self.Coef = [Coef_0, Coef_1, Coef_2]
        self.z_ssc = []
        cost_ssc = 0
        reg_ssc = 0

        # Encoder
        H = X
        self.H_in = []
        self.H_in.append(H)
        for layer in range(self.n_layers):
            H = self.__encoder(A, prune_A, H, layer)
            if self.nonlinear:
                if layer != self.n_layers-1:
                    H = tf.nn.elu(H)
            self.H_in.append(H)
        # Final node representations
        self.H = H
        self.z = self.H

        # append C * H
        for i in range(self.n_layers + 1):
            z_ssc = tf.matmul(self.Coef[i], self.H_in[i])
            self.z_ssc.append(z_ssc)

        # Coef-fushion
        self.Coef_mean = 1 / (weight_0 + weight_1 + weight_2) * (
                    weight_0 * Coef_0 + weight_1 * Coef_1 + weight_2 * Coef_2)

        # Decoder
        self.H_out_1 = []
        H = tf.matmul(self.Coef_mean, self.z)
        self.H_out_1.append(H)
        for layer in range(self.n_layers - 1, -1, -1):
            H = self.__decoder(H, layer)
            if self.nonlinear:
                if layer != 0:
                    H = tf.nn.elu(H)
            self.H_out_1.append(H)
        self.H_out_1.reverse()
        X_ = H
        
        # The reconstruction loss of node features
        #features_loss = tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.pow(X - X_, 2))))
        features_loss = 1 / 2 * (tf.reduce_sum(tf.reduce_sum(tf.pow(X - X_, 2))))
        weight_decay_loss = 0
        for layer in range(self.n_layers):

            weight_decay_loss += tf.multiply(tf.nn.l2_loss(self.W[layer]), self.weight_decay, name='weight_loss')

        # self expression loss ( * 6)
        for i in range(self.n_layers + 1):
            cost_ssc = 1 / 2 * (tf.reduce_sum(tf.pow(tf.subtract(self.H_in[i], self.z_ssc[i]), 2)))
            reg_ssc = tf.reduce_mean(tf.pow(self.Coef[i], 2))  # reduce_sum ?
            cost_ssc += cost_ssc
            reg_ssc += reg_ssc
        self.reg_ssc = 1/len(self.hidden_dims) * self.reg_ssc_coef * reg_ssc
        self.cost_ssc = 1/len(self.hidden_dims) * self.cost_ssc_coef * cost_ssc

        self.features_loss = 1.0 * features_loss
        self.reg_ssc_loss = 1.0 * self.reg_ssc
        self.cost_ssc_loss = 1.0 * self.cost_ssc

        # self supervised
        Pseudo_L = self.__dense_layer(self.z)
        self.Clustering_results = tf.placeholder(dtype=tf.float32)
        L_self_supervised = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=Pseudo_L, logits=self.Clustering_results))
        self.L_self_supervised_loss = 1.0 * self.L_self_supervised_coef * L_self_supervised

        # Total loss
        #self.loss = self.features_loss+ weight_decay_loss

        self.loss = self.features_loss + self.reg_ssc_loss + self.cost_ssc_loss \
                    + weight_decay_loss+ self.L_self_supervised_loss

        if self.alpha == 0:
            self.Att_l = self.C
        else:
            #self.Att_l = {x: (1-self.alpha)*self.C[x] + self.alpha*self.prune_C[x] for x in self.C.keys()}
            self.Att_l = {'C': self.C, 'prune_C': self.prune_C}
        return self.loss, self.H, self.Att_l, self.z, self.Coef_mean, self.Clustering_results \
            , self.features_loss, self.reg_ssc_loss, self.cost_ssc_loss \
            , self.L_self_supervised_loss,X_

    def __dense_layer(self, Z):
        dense1 = tf.layers.dense(inputs=Z, units=128, activation=tf.nn.relu)
        dense2 = tf.layers.dense(inputs=dense1, units=64, activation=tf.nn.relu)
        dense3 = tf.layers.dense(inputs=dense2, units=32, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense3, units=self.n_cluster, activation=None)
        return logits

    def __encoder(self, A, prune_A, H, layer):
        H = tf.matmul(H, self.W[layer])
        if layer == self.n_layers-1:
            return H
        self.C[layer] = self.graph_attention_layer(A, H, self.v[layer], layer)
        if self.alpha == 0:
            return tf.sparse_tensor_dense_matmul(self.C[layer], H)
        else:
            self.prune_C[layer] = self.graph_attention_layer(prune_A, H, self.prune_v[layer], layer)
            return (1-self.alpha)*tf.sparse_tensor_dense_matmul(self.C[layer], H) + self.alpha*tf.sparse_tensor_dense_matmul(self.prune_C[layer], H)


    def __decoder(self, H, layer):
        H = tf.matmul(H, self.W[layer], transpose_b=True)
        if layer == 0:
            return H
        if self.alpha == 0:
            return tf.sparse_tensor_dense_matmul(self.C[layer-1], H)
        else:
            return (1-self.alpha)*tf.sparse_tensor_dense_matmul(self.C[layer-1], H) + self.alpha*tf.sparse_tensor_dense_matmul(self.prune_C[layer-1], H)


    def define_weights(self, hidden_dims):
        W = {}
        for i in range(self.n_layers):
            W[i] = tf.get_variable("W%s" % i, shape=(hidden_dims[i], hidden_dims[i+1]))

        Ws_att = {}
        for i in range(self.n_layers-1):
            v = {}
            v[0] = tf.get_variable("v%s_0" % i, shape=(hidden_dims[i+1], 1))
            v[1] = tf.get_variable("v%s_1" % i, shape=(hidden_dims[i+1], 1))

            Ws_att[i] = v
        if self.alpha == 0:
            return W, Ws_att, None
        prune_Ws_att = {}
        for i in range(self.n_layers-1):
            prune_v = {}
            prune_v[0] = tf.get_variable("prune_v%s_0" % i, shape=(hidden_dims[i+1], 1))
            prune_v[1] = tf.get_variable("prune_v%s_1" % i, shape=(hidden_dims[i+1], 1))

            prune_Ws_att[i] = prune_v

        return W, Ws_att, prune_Ws_att

    def graph_attention_layer(self, A, M, v, layer):

        with tf.variable_scope("layer_%s"% layer):
            f1 = tf.matmul(M, v[0])
            f1 = A * f1
            f2 = tf.matmul(M, v[1])
            f2 = A * tf.transpose(f2, [1, 0])
            logits = tf.sparse_add(f1, f2)

            unnormalized_attentions = tf.SparseTensor(indices=logits.indices,
                                         values=tf.nn.sigmoid(logits.values),
                                         dense_shape=logits.dense_shape)
            attentions = tf.sparse_softmax(unnormalized_attentions)

            attentions = tf.SparseTensor(indices=attentions.indices,
                                         values=attentions.values,
                                         dense_shape=attentions.dense_shape)

            return attentions

    def post_proC(self, C, K, d=6, alpha=8):
        # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
        C = 0.5 * (C + C.T)
        r = d * K + 1
        U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
        U = U[:, ::-1]
        S = np.sqrt(S[::-1])
        S = np.diag(S)
        U = U.dot(S)
        U = normalize(U, norm='l2', axis=1)
        Z = U.dot(U.T)
        Z = Z * (Z > 0)
        L = np.abs(Z ** alpha)
        L = L / L.max()
        L = 0.5 * (L + L.T)
        spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                              assign_labels='discretize')
        spectral.fit(L)
        grp = spectral.fit_predict(L)
        return grp, L

    def thrC(self, C, ro):
        if ro < 1:
            N = C.shape[1]
            Cp = np.zeros((N, N))
            S = np.abs(np.sort(-np.abs(C), axis=0))
            Ind = np.argsort(-np.abs(C), axis=0)
            for i in range(N):
                cL1 = np.sum(S[:, i]).astype(float)
                stop = False
                csum = 0
                t = 0
                while stop == False:
                    csum = csum + S[t, i]
                    if csum > ro * cL1:
                        stop = True
                        Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                    t = t + 1
        else:
            Cp = C

        return Cp
