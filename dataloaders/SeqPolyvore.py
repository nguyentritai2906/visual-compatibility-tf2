import json

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras.utils import Sequence


class SeqPolyvore(Sequence):

    def __init__(self,
                 datapath,
                 phase,
                 support_dropout_rate,
                 degree,
                 adj_self_connection,
                 mean=None,
                 std=None):
        assert phase in ['train', 'valid', 'test']
        self.phase = phase
        self.datapath = datapath
        self.support_dropout_rate = support_dropout_rate
        self.degree = degree
        self.adj_self_connection = adj_self_connection

        adj_file = datapath + 'adj_{}.npz'.format(phase)
        feats_file = datapath + 'features_{}.npz'.format('train')
        np.random.seed(1234)

        adj = sp.load_npz(adj_file).astype(np.int32)
        # get lower tiangle of the adj matrix to avoid duplicate edges
        self.lower_adj = sp.tril(adj).tocsr()
        # self.{train/val/test}_features
        self.norm_features = self.normalize_features(sp.load_npz(feats_file),
                                                     mean, std)

        questions_file = datapath + 'questions_test.json'
        questions_file_resampled = questions_file.replace(
            'questions', 'questions_RESAMPLED')
        with open(questions_file) as f:
            self.questions = json.load(f)
        with open(questions_file_resampled) as f:
            self.questions_resampled = json.load(f)

        self.support, self.labels, self.r_idx, self.c_idx = self.get_phase()

        self.num_support = len(self.support)
        self.on_epoch_end()

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if self.support_dropout_rate and self.phase == 'train':
            dropout_support = []
            dropout_support.append(self.support[0])
            # do not modify the first support, the self-connections one
            for i in range(1, len(self.support)):
                modified = self.support_dropout(sup=self.support[i],
                                                do=self.support_dropout_rate,
                                                edge_drop=True)
                modified.data[...] = 1  # make it binary to normalize
                modified = self.normalize_nonsym_adj(modified)
                dropout_support.append(modified)

            dropout_support = [
                self.to_sparse_tensor(s) for s in dropout_support
            ]
            dropout_support = tf.sparse.concat(axis=0,
                                               sp_inputs=dropout_support)

        else:
            dropout_support = self.support.copy()
            dropout_support = [
                self.to_sparse_tensor(s) for s in dropout_support
            ]
            dropout_support = tf.sparse.concat(axis=0,
                                               sp_inputs=dropout_support)

        return (self.norm_features, dropout_support, self.r_idx,
                self.c_idx), tf.cast(self.labels, tf.float32)

    def get_phase(self):
        # get the positive edges
        pos_r_idx, pos_c_idx = self.lower_adj.nonzero()
        pos_labels = np.array(self.lower_adj[pos_r_idx, pos_c_idx]).squeeze()

        # split the positive edges into the ones used for evaluation and the ones used as message passing
        n_pos = pos_labels.shape[0]  # number of positive edges
        perm = list(range(n_pos))
        np.random.shuffle(perm)
        pos_labels, pos_r_idx, pos_c_idx = pos_labels[perm], pos_r_idx[
            perm], pos_c_idx[perm]
        n_eval = int(n_pos / 2)
        # message passing edges
        mp_pos_labels, mp_pos_r_idx, mp_pos_c_idx = pos_labels[
            n_eval:], pos_r_idx[n_eval:], pos_c_idx[n_eval:]
        # this are the positive examples that will be used to compute the loss function
        eval_pos_labels, eval_pos_r_idx, eval_pos_c_idx = pos_labels[:
                                                                     n_eval], pos_r_idx[:
                                                                                        n_eval], pos_c_idx[:
                                                                                                           n_eval]

        # get the negative edges
        # set the number of negative training edges that will be needed to sample at each iter
        n_train_neg = eval_pos_labels.shape[0]
        neg_labels = np.zeros((n_train_neg))
        # get the possible indexes to be sampled (basically all indexes if there aren't restrictions)
        poss_nodes = np.arange(self.lower_adj.shape[0])

        neg_r_idx = np.zeros((n_train_neg))
        neg_c_idx = np.zeros((n_train_neg))

        for i in range(n_train_neg):
            r_idx, c_idx = self.get_negative_training_edge(
                poss_nodes.shape[0], self.lower_adj)
            neg_r_idx[i] = r_idx
            neg_c_idx[i] = c_idx

        # build adj matrix
        adj = sp.csr_matrix(
            (np.hstack([mp_pos_labels, mp_pos_labels]),
             (np.hstack([mp_pos_r_idx, mp_pos_c_idx
                         ]), np.hstack([mp_pos_c_idx, mp_pos_r_idx]))),
            shape=(self.lower_adj.shape[0], self.lower_adj.shape[0]))
        # remove the labels of the negative edges which are 0
        adj.eliminate_zeros()

        labels = np.append(eval_pos_labels, neg_labels)
        r_idx = np.append(eval_pos_r_idx, neg_r_idx)
        c_idx = np.append(eval_pos_c_idx, neg_c_idx)

        support = self.get_degree_supports(adj, self.degree,
                                           self.adj_self_connection)
        for i in range(1, len(support)):
            support[i] = self.normalize_nonsym_adj(support[i])

        return support, labels, r_idx, c_idx

    def normalize_features(self, features, mean=None, std=None):
        reuse_mean = mean is not None and std is not None
        if features.shape[1] != 2048:  # image features
            raise ValueError(
                'Features are expected to be 2048-dimensional (extracted with ResNet)'
            )
        else:
            features = np.array(features.todense())
            if reuse_mean:
                self.mean_feats = mean
                self.std_feats = std
            else:
                self.mean_feats = features.mean(axis=0)
                self.std_feats = features.std(axis=0)

            # normalize
            norm_features = (features - self.mean_feats) / self.std_feats
        return norm_features

    def get_moments(self):
        return self.mean_feats, self.std_feats

    def get_test_questions(self, resampled=False):
        """
        Return the FITB 'questions' in form of node indexes
        """
        # each question consists on N*4 edges to predict
        # self.questions is a list of questions with N elements and 4 possible choices (answers)
        flat_questions = []
        gt = []
        # this list indicates to which question does each edge belongs to
        q_ids = []
        valid = []
        q_id = 0
        questions = self.questions if not resampled else self.questions_resampled
        for question in questions:
            # indexes of outfit nodes
            for index in question[0]:
                i = 0
                # indexes of possible choices answers
                for index_answer in question[1]:
                    flat_questions.append([index,
                                           index_answer])  # append the edge
                    if i == 0:
                        gt.append(1)  # the correct connection is the first
                    else:
                        gt.append(0)
                    # a link is valid if the candidate item is from the same category as the missing item
                    valid.append(int(question[2][i] == question[3]))
                    i += 1
                    q_ids.append(q_id)
            q_id += 1

        assert len(flat_questions) == len(gt) and len(q_ids) == len(
            gt) and len(gt) == len(valid)
        assert len(self.questions) == max(q_ids) + 1

        # flat questions contains edges [u,v]
        # gt contains the ground truth label for each edge
        # q_ids indicates to which question does each query edge belong to

        flat_questions = np.array(flat_questions)
        gt = np.array(gt)
        q_ids = np.array(q_ids)
        valid = np.array(valid)

        # now build the adj for message passing for the questions, by removing the edges that will be evaluated
        lower_adj = self.lower_adj.copy()

        full_adj = lower_adj + lower_adj.transpose()
        full_adj = full_adj.tolil()
        for edge, label in zip(flat_questions, gt):
            u, v = edge
            full_adj[u, v] = 0
            full_adj[v, u] = 0

        full_adj = full_adj.tocsr()
        full_adj.eliminate_zeros()

        # make sure that none of the query edges are in the adj matrix
        count_edges = 0
        count_pos = 0
        for edge in flat_questions:
            u, v = edge
            if full_adj[u, v] > 0:
                count_pos += 1
            count_edges += 1
        assert count_pos == 0

        return full_adj, flat_questions[:,
                                        0], flat_questions[:,
                                                           1], gt, q_ids, valid

    def normalize_nonsym_adj(self, adj):
        degree = np.asarray(adj.sum(1)).flatten()

        # set zeros to inf to avoid dividing by zero
        degree[degree == 0.] = np.inf

        degree_inv_sqrt = 1. / np.sqrt(degree)
        degree_inv_sqrt_mat = sp.diags([degree_inv_sqrt], [0])

        degree_inv = degree_inv_sqrt_mat.dot(degree_inv_sqrt_mat)

        adj_norm = degree_inv.dot(adj)

        return adj_norm

    def to_sparse_tensor(self, X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data.astype(np.float32), coo.shape)

    def get_degree_supports(self, adj, k, adj_self_con=False, verbose=True):
        if verbose:
            print('Computing adj matrices up to {}th degree'.format(k))
        supports = [sp.identity(adj.shape[0])]
        if k == 0:  # return Identity matrix (no message passing)
            return supports
        assert k > 0
        supports = [
            sp.identity(adj.shape[0]),
            adj.astype(np.float64) + adj_self_con * sp.identity(adj.shape[0])
        ]

        prev_power = adj
        for _ in range(k - 1):
            power = prev_power.dot(adj)
            new_adj = ((power) == 1).astype(np.float64)
            new_adj.setdiag(0)
            new_adj.eliminate_zeros()
            supports.append(new_adj)
            prev_power = power
        return supports

    def support_dropout(self, sup, do, edge_drop=False):
        sup = sp.tril(sup)
        assert do > 0.0 and do < 1.0
        n_nodes = sup.shape[0]
        # nodes that I want to isolate
        isolate = np.random.choice(range(n_nodes),
                                   int(n_nodes * do),
                                   replace=False)
        nnz_rows, nnz_cols = sup.nonzero()

        # mask the nodes that have been selected
        mask = np.in1d(nnz_rows, isolate)
        mask += np.in1d(nnz_cols, isolate)
        assert mask.shape[0] == sup.data.shape[0]

        sup.data[mask] = 0
        sup.eliminate_zeros()

        if edge_drop:
            prob = np.random.uniform(0, 1, size=sup.data.shape)
            remove = prob < do
            sup.data[remove] = 0
            sup.eliminate_zeros()

        sup = sup + sup.transpose()
        return sup

    def get_negative_training_edge(self, num_nodes, lower_adj):
        u = np.random.randint(num_nodes)
        v = np.random.randint(num_nodes)

        while lower_adj[u, v] or lower_adj[v, u]:  # sampled a positive edge
            u = np.random.randint(num_nodes)

        return u, v
