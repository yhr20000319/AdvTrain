import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import tensorflow as tf2
tf = tf2.compat.v1
tf.disable_v2_behavior()

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation.
    特征矩阵需要使用占位符placeholder传入模型内部，而邻接矩阵是全局共享不变的不需要占位符，
    而稀疏站位符tf.sparse_placeholder的格式是（行列索引，值，shape）和coo_matrix对应，
    因此代码中最后转化为coo_matrix
    """

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation
    # 将特征从稀疏矩阵，行归一化之后，转化成tuple格式"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)  # 返回一个稀疏矩阵的非0值坐标、非0值和整个矩阵的shape
    # 这个coo_matrix类型 其实就是系数矩阵的坐标形式：（所有非0元素 （row，col））根据row和col对应的索引对应非0元素在矩阵中的位置
    # 其他位置自动补0


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix.
    对称归一化邻接矩阵。 对称归一化 D-0.5*A*D-0.5"""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation.
    简单GCN模型邻接矩阵的预处理及元组表示的转换 adj是sparse  """
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders, train=False):
    """Construct feed dictionary.
    construct_feed_dict拿到了在train.py定义的placeholders，placeholders
    拿到指定的key替换为placeholders中的value（各种tensorflow tensor对象）作为key，
    以具体的值作为value，装进feat_dict中

    """
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})

    if train:
        feed_dict.update(
            {placeholders['support'][i]: support[i] for i in range(len(support))})  # if attack: do not feed in support
        feed_dict.update(
            {placeholders['ori_support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


#  限制函数，限定在规定扰动率内，eps扰动限制的总数量
def bisection(a, eps, xi, ub=1):
    pa = np.clip(a, 0, ub)  # 截取函数，数组a中的所有数限定到范围a_min=0和a_max=ub=1中
    if np.sum(pa) <= eps:
        # print('np.sum(pa) <= eps !!!!')
        upper_S_update = pa
    else:  # 不在扰动范围内，就用二分查找找到一个合适的值，减去后让其在范围内
        mu_l = np.min(a - 1)
        mu_u = np.max(a)
        # mu_a = (mu_u + mu_l)/2
        while np.abs(mu_u - mu_l) > xi:
            # print('|mu_u - mu_l|:',np.abs(mu_u - mu_l))
            mu_a = (mu_u + mu_l) / 2
            gu = np.sum(np.clip(a - mu_a, 0, ub)) - eps
            gu_l = np.sum(np.clip(a - mu_l, 0, ub)) - eps
            # print('gu:',gu)
            if gu == 0:
                print('gu == 0 !!!!!')
                break
            if np.sign(gu) == np.sign(gu_l):
                mu_l = mu_a
            else:
                mu_u = mu_a

        upper_S_update = np.clip(a - mu_a, 0, ub)

    return upper_S_update


def filter_potential_singletons(adj):
    """
    Computes a mask for entries potentially leading to singleton nodes, i.e. one of the two nodes corresponding to
    the entry have degree 1 and there is an edge between the two nodes.
    Returns
    -------
    tf.Tensor shape [N, N], float with ones everywhere except the entries of potential singleton nodes,
    where the returned tensor has value 0.
    """
    adj = np.squeeze(adj)
    N = adj.shape[-1]
    degrees = np.sum(adj, axis=0)
    degree_one = np.equal(degrees, 1)
    resh = np.reshape(np.tile(degree_one, [N]), [N, N])
    l_and = np.logical_and(resh, np.equal(adj, 1))
    logical_and_symmetric = np.logical_or(l_and, np.transpose(l_and))
    return logical_and_symmetric

