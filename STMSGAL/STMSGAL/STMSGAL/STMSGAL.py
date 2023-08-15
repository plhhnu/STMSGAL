import tensorflow as tf2
tf = tf2.compat.v1
tf.disable_v2_behavior()
import scipy.sparse as sp
import numpy as np
from .model import GATE
from tqdm import tqdm
import matplotlib.pyplot as plt

class STMSGAL():

    def __init__(self, hidden_dims, alpha, n_epochs=500, lr=0.0001, 
                 gradient_clipping=5, nonlinear=True, weight_decay=0.0001, 
                 verbose=True, random_seed=2020,batchsize = 0,n_cluster = 6,
                 reg_ssc_coef=1.0,cost_ssc_coef=1.0,L_self_supervised_coef = 1.0,
                 category = "dataset"):
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
        self.category = category
        self.loss_list = []
        self.lr = lr
        self.n_epochs = n_epochs
        self.gradient_clipping = gradient_clipping
        self.build_placeholders()
        self.verbose = verbose
        self.alpha = alpha
        self.batch_size = batchsize
        self.n_cluster =n_cluster
        self.gate = GATE(hidden_dims, alpha, nonlinear, weight_decay,batchsize,self.n_cluster,
                         reg_ssc_coef, cost_ssc_coef,L_self_supervised_coef)
        self.loss, self.H, self.C, self.z, self.Coef_mean, self.Clustering_results, self.features_loss \
            ,  self.reg_ssc_loss, self.cost_ssc_loss \
            , self.L_self_supervised_loss, self.ReX= self.gate(self.A, self.prune_A, self.X)
        self.optimize(self.loss)
        self.build_session()
        self.pred_dsc = []

    def build_placeholders(self):
        self.A = tf.sparse_placeholder(dtype=tf.float32)
        self.prune_A = tf.sparse_placeholder(dtype=tf.float32)
        self.X = tf.placeholder(dtype=tf.float32)

    def build_session(self, gpu= True):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if gpu == False:
            config.intra_op_parallelism_threads = 0
            config.inter_op_parallelism_threads = 0
        self.session = tf.Session(config=config)
        self.session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    def optimize(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clipping)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

    def __call__(self, A, prune_A, X):
        list_loss = []
        for epoch in tqdm(range(self.n_epochs)):
            total_loss,pred_dsc = self.run_epoch(epoch, A, prune_A, X)
            list_loss.append(total_loss)
        self.pred_dsc = pred_dsc

        self.loss_list = list_loss
        x = np.arange(self.n_epochs)
        # plt.rcParams["figure.figsize"] = (5, 5)
        # plt.plot(x, list_loss, label='total_loss')
        # plt.legend(['total_loss'])
        # plt.title(f"")
        # plt.savefig(f'./outputss/{self.category}_epoch{self.n_epochs}.jpg',bbox_inches='tight', dpi=150)
        # plt.show()
        # print("plt finish")

    def run_epoch(self, epoch, A, prune_A, X):
        coef_mean = self.session.run(self.Coef_mean,
                                     feed_dict={self.A: A,
                                                self.X: X,
                                                # self.S: S,
                                                # self.R: R,
                                                # self.p: p
                                                })
        alpha = max(0.4 - (self.n_cluster - 1) / 10 * 0.1, 0.1)
        commonZ = self.gate.thrC(coef_mean, alpha)
        pred_dsc, _ = self.gate.post_proC(commonZ, self.n_cluster)
        pred_dsc_encoded = np.zeros((self.batch_size, self.n_cluster))
        for i in range(self.batch_size):
            pred_dsc_encoded[i][pred_dsc[i]] = 1

        total_loss, features_loss, reg_ssc_loss, cost_ssc_loss, l_self_supervised_loss, _ \
            = self.session.run([self.loss, self.features_loss, self.reg_ssc_loss
                                   , self.cost_ssc_loss, self.L_self_supervised_loss, self.train_op],
                               feed_dict={self.A: A,
                                          self.prune_A: prune_A,
                                          self.X: X,
                                          # self.S: S,
                                          # self.R: R,
                                          # self.p: p
                                          self.Clustering_results: pred_dsc_encoded
                                          })

        # loss, _ = self.session.run([self.loss, self.train_op],
        #                                  feed_dict={self.A: A,
        #                                             self.prune_A: prune_A,
        #                                             self.X: X})
        self.loss_list.append(total_loss)
        print("epoch: {}\ttotal_loss: {}\t".format(
                epoch, total_loss))
        print("features: {}\treg_c: {}\tcost_c: {}\tself_supervised: {}\n"
            .format(features_loss, reg_ssc_loss, cost_ssc_loss, l_self_supervised_loss))

        #if self.verbose:
        #    print("Epoch: %s, Loss: %.4f" % (epoch, loss))
        return total_loss,pred_dsc

    def infer(self, A, prune_A, X):
        H, C, ReX = self.session.run([self.H, self.C, self.ReX],
                           feed_dict={self.A: A,
                                      self.prune_A: prune_A,
                                      self.X: X})

        return H, self.Conbine_Atten_l(C), self.loss_list, ReX,self.pred_dsc

    def Conbine_Atten_l(self, input):
        if self.alpha == 0:
            return [sp.coo_matrix((input[layer][1], (input[layer][0][:, 0], input[layer][0][:, 1])), shape=(input[layer][2][0], input[layer][2][1])) for layer in input]
        else:
            Att_C = [sp.coo_matrix((input['C'][layer][1], (input['C'][layer][0][:, 0], input['C'][layer][0][:, 1])), shape=(input['C'][layer][2][0], input['C'][layer][2][1])) for layer in input['C']]
            Att_pruneC = [sp.coo_matrix((input['prune_C'][layer][1], (input['prune_C'][layer][0][:, 0], input['prune_C'][layer][0][:, 1])), shape=(input['prune_C'][layer][2][0], input['prune_C'][layer][2][1])) for layer in input['prune_C']]
            return [self.alpha*Att_pruneC[layer] + (1-self.alpha)*Att_C[layer] for layer in input['C']]
