# main.py


# dependencies
import os
import uuid
import glob
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import jsonlines
from math import ceil
from torch.autograd import Variable
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# set up path
cwd = os.getcwd()
data_path = os.path.join(cwd, 'aug_data')
train_path = os.path.join(cwd, 'train_data')
valid_path = os.path.join(cwd, 'valid_data')
total_path = os.path.join(cwd, 'total_data')

# TODO: download aug data
# TODO: package


# set up device
# If there's a GPU available...
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')
print('-'* 30)


# set random seed
seed = 4747
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# load data
aug_token_embeddings = np.load(os.path.join(data_path, 'wordvector_masked_aug.npy'))
token_embeddings = np.load(os.path.join(data_path, 'train_wordvectors.npy'))
train_pos_positions = pickle.load(open(os.path.join(data_path, 'train_pos_dict.pkl'), 'rb'))
train_neg_positions = pickle.load(open(os.path.join(data_path, 'train_neg_dict.pkl'), 'rb'))
valid_pos_positions = pickle.load(open(os.path.join(data_path, 'val_pos_dict.pkl'), 'rb'))
valid_neg_positions = pickle.load(open(os.path.join(data_path, 'val_neg_dict.pkl'), 'rb'))
print('Data Loaded')
print('-'* 30)


# data functions
# data loader
class CL_Data_loader():
    # data is a list with all feature arrays
    # X_train_pos, X_train_neg, X_val_pos,X_val_neg only contain indexes for train and validation
    # augmentation degree: alpha [0(none), 1, 2, 3, 4, 5],
    # augmention method [1-4], insert, swap, delete, replace
    # No Augmentation: alpha 0, method 0
    # data: original data
    # aug_data: augmented data in the format of array, i.e. aug_data[alpha][method]

    def __init__(self, X_train_pos, X_train_neg, X_val_pos,X_val_neg,
                 data, aug_data, batch_size, k_shot=1, train_mode=True):

        self.data = data
        self.aug_data = aug_data

        self.batch_size = batch_size

        self.k_shot = k_shot # 1 or 5, how many times the model sees the example

        self.num_classes = 2   # this is a binary classification

        self.train_pos = X_train_pos
        self.train_neg = X_train_neg

        # position of last batch
        self.train_pos_index = 0
        self.train_neg_index = 0

        if not train_mode:

            self.val_pos = X_val_pos
            self.val_neg = X_val_neg

            self.val_pos_index = 0
            self.val_neg_index = 0

            # merge train & val for prediction use
            self.all_pos = np.concatenate([self.train_pos, self.val_pos])
            self.all_neg = np.concatenate([self.train_neg, self.val_neg])

            self.pos_index = 0
            self.neg_index = 0


        self.iters = 100


    def next_batch(self, alpha = 0, aug_type = 0, return_sample_ids=False):
        x_set_batch = []
        y_set_batch = []
        x_hat_batch = []
        y_hat_batch = []


        x_set = []
        y_set = []

        for _ in range(self.batch_size):

            x_set = []
            y_set = []

            target_class = np.random.randint(self.num_classes)
            #print(target_class)

            # negative class
            for i in range(self.k_shot+1):

                # shuffle pos or neg if a sequence has been full used
                if self.train_neg_index == len(self.train_neg):

                    self.train_neg = np.random.permutation(self.train_neg)
                    self.train_neg_index = 0
                    #print("neg seq", self.train_neg_seq)

                if i==self.k_shot:  # the last one is test sample

                    if target_class == 0: # positive class
                        x_hat_batch.append(self.train_neg[self.train_neg_index])

                        y_hat_batch.append(0)
                        self.train_neg_index += 1
                else:

                    x_set.append(self.train_neg[self.train_neg_index])

                    y_set.append(0)
                    self.train_neg_index += 1

            # positive class
            for i in range(self.k_shot+1):

                # shuffle pos or neg if a sequence has been full used
                if self.train_pos_index == len(self.train_pos):

                    self.train_pos = np.random.permutation(self.train_pos)
                    self.train_pos_index = 0
                    #print("pos seq", self.train_pos_seq)

                if i==self.k_shot:  # the last one is test sample

                    if target_class == 1: # positive class
                        x_hat_batch.append(self.train_pos[self.train_pos_index])

                        y_hat_batch.append(1)
                        self.train_pos_index += 1


                else:
                    x_set.append(self.train_pos[self.train_pos_index])

                    y_set.append(1)
                    self.train_pos_index += 1

            x_set_batch.append(x_set)
            y_set_batch.append(y_set)

            # get feature arrays for the batch

        #print(x_set_batch)
        #print(x_hat_batch)

        feature_set_batch = []
        feature_hat_batch = []

        for did, feature in enumerate(self.data):
            if did == 0:    # word vector

                if alpha == 0:  # no augmentation
                    f_set = np.array([np.array(feature[b]) for b in x_set_batch])
                    f_hat = np.array(feature[x_hat_batch])
                else:
                    feature = self.aug_data[0][alpha - 1][aug_type - 1]
                    f_set = np.array([np.array(feature[b]) for b in x_set_batch])
                    f_hat = np.array(feature[x_hat_batch])

            else:
                f_set = np.array([np.array(feature[b]) for b in x_set_batch])
                f_hat = np.array(feature[x_hat_batch])

            # reshape support to (batch, n_way, k_shot, *feature size)
            f_set = f_set.reshape((self.batch_size, 2, self.k_shot, *(feature.shape[1:])))
            #print(f_set.shape)
            #print(f_hat.shape)

            feature_set_batch.append(f_set)
            feature_hat_batch.append(f_hat)

        feature_set_batch = self.convert_to_tensor(feature_set_batch)
        feature_hat_batch = self.convert_to_tensor(feature_hat_batch)
        y_hat_batch = torch.Tensor(np.asarray(y_hat_batch).astype(np.int32))

        if return_sample_ids:
            return feature_set_batch, feature_hat_batch, y_hat_batch, x_set_batch, x_hat_batch  # sample IDs
        else:
            return feature_set_batch, feature_hat_batch, y_hat_batch
            #np.zeros(self.batch_size)   # all 0s for aux output


    def next_eval_batch(self, return_sample_ids = False):
        x_set_batch = []
        y_set_batch = []
        x_hat_batch = []
        y_hat_batch = []

        for _ in range(self.batch_size):

            x_set = []
            y_set = []

            target_class = np.random.randint(self.num_classes)
            #print(target_class)

            if self.val_pos_index == len(self.val_pos):
                self.val_pos = np.random.permutation(self.val_pos)
                self.val_pos_index = 0
                #print("pos val seq", self.val_pos_seq)

            if self.val_neg_index == len(self.val_neg):
                self.val_neg = np.random.permutation(self.val_neg)
                self.val_neg_index = 0
                #print("net val seq", self.val_neg_seq)

            # negative class
            for i in range(self.k_shot+1):

                # shuffle pos or neg if a sequence has been full used
                if self.train_neg_index == len(self.train_neg):
                    self.train_neg = np.random.permutation(self.train_neg)
                    self.train_neg_index = 0
                    #print("neg seq", self.train_neg_seq)

                if i==self.k_shot:  # the last one is test sample

                    if target_class == 0: # negative class
                        x_hat_batch.append(self.val_neg[self.val_neg_index])

                        y_hat_batch.append(0)
                        self.val_neg_index += 1
                else:

                    x_set.append(self.train_neg[self.train_neg_index])

                    y_set.append(0)
                    self.train_neg_index += 1

            # positive class
            for i in range(self.k_shot+1):

                # shuffle pos or neg if a sequence has been full used
                if self.train_pos_index == len(self.train_pos):
                    self.train_pos = np.random.permutation(self.train_pos)
                    self.train_pos_index = 0
                    #print("pos seq", self.train_pos_seq)

                if i==self.k_shot:  # the last one is test sample

                    if target_class == 1: # positive class
                        x_hat_batch.append(self.val_pos[self.val_pos_index])

                        y_hat_batch.append(1)
                        self.val_pos_index += 1

                else:
                    x_set.append(self.train_pos[self.train_pos_index])

                    y_set.append(1)
                    self.train_pos_index += 1

            x_set_batch.append(x_set)
            y_set_batch.append(y_set)

        #print(x_set_batch)
        #print(x_hat_batch)

        feature_set_batch = []
        feature_hat_batch = []

        # loop through all features
        for feature in self.data:
            f_set = np.array([np.array(feature[b]) for b in x_set_batch])
            f_hat = np.array(feature[x_hat_batch])
            #print(f_set.shape)
            #print(f_hat.shape)

            f_set = f_set.reshape((self.batch_size, 2, self.k_shot, *(feature.shape[1:])))

            feature_set_batch.append(f_set)
            feature_hat_batch.append(f_hat)

        if return_sample_ids:
            return feature_set_batch, np.asarray(y_set_batch).astype(np.int32), feature_hat_batch, np.asarray(y_hat_batch).astype(np.int32), x_set_batch, x_hat_batch  # get sample IDs
        else:
            return feature_set_batch, np.asarray(y_set_batch).astype(np.int32), feature_hat_batch, np.asarray(y_hat_batch).astype(np.int32)

    # generate support set for each sample in prediction
    # use all samples as support
    def get_pred_set(self, pred, return_sample_ids=False):
        x_set_batch = []
        y_set_batch = []
        x_hat_batch = []

        for _ in range(self.batch_size):  #batch_size = 32

            x_set = []
            y_set = []

            target_class = np.random.randint(self.num_classes)   #target_class = 0/1
            #print(target_class)

            if self.pos_index == len(self.all_pos):  #initiate the index
                self.all_pos = np.random.permutation(self.all_pos)
                self.pos_index = 0

            if self.neg_index == len(self.all_neg):   #initiate the index
                self.all_neg = np.random.permutation(self.all_neg)
                self.neg_index = 0

            # negative class
            for i in range(self.k_shot):

                # shuffle pos or neg if a sequence has been full used
                if self.neg_index == len(self.all_neg):
                    self.all_neg = np.random.permutation(self.all_neg)
                    self.neg_index = 0
                    #print("neg seq", self.train_neg_seq)

                x_set.append(self.all_neg[self.neg_index])

                y_set.append(0)
                self.neg_index += 1

            # positive class
            for i in range(self.k_shot):

                # shuffle pos or neg if a sequence has been full used
                if self.pos_index == len(self.all_pos):
                    self.all_pos = np.random.permutation(self.all_pos)
                    self.pos_index = 0
                    #print("pos seq", self.train_pos_seq)

                x_set.append(self.all_pos[self.pos_index])

                y_set.append(1)
                self.pos_index += 1

            # Prediction sample

            x_set_batch.append(x_set)
            y_set_batch.append(y_set)

        x_hat_batch.append(pred)
        #print(x_set_batch)

        # repeat each element in pred for batch_size times
        feature_hat_batch = [np.repeat(e[None,:], self.batch_size, axis = 0) for e in pred]
        feature_set_batch = []

        # loop through all features
        for idx, feature in enumerate(self.data):
            f_set = np.array([np.array(feature[b]) for b in x_set_batch])
            #print(f_set.shape)

            f_set = f_set.reshape((self.batch_size, 2, self.k_shot, *(feature.shape[1:])))

            feature_set_batch.append(f_set)

        if return_sample_ids:
            return feature_set_batch, np.asarray(y_set_batch).astype(np.int32), feature_hat_batch, x_set_batch
        else:
            return feature_set_batch, np.asarray(y_set_batch).astype(np.int32), feature_hat_batch

    #def get_pred_set_gen(self, pred):
    #    while True:
    #        x_set, y_set, x_hat, y_hat = train_loader.next_batch()
    #        yield([x_set, x_hat], 1-y_hat)

    def convert_to_tensor(self, features):
        tensors = [torch.Tensor(item) for item in features]
        return tensors


    def next_eval_batch_gen(self, return_sample_ids=False):
        while True:
            if return_sample_ids:
                x_set, y_set, x_hat, y_hat, x_set_ids, x_hat_ids = self.next_eval_batch(return_sample_ids = return_sample_ids)
            else:
                x_set, y_set, x_hat, y_hat = self.next_eval_batch(return_sample_ids = return_sample_ids)

            x_set = self.convert_to_tensor(x_set)
            x_hat = self.convert_to_tensor(x_hat)
            y_hat = torch.Tensor(y_hat)

            if return_sample_ids:
                yield (x_set, x_hat, y_hat, x_set_ids, x_hat_ids)
            else:
                yield(x_set, x_hat, y_hat)



    def next_batch_by_ids(self, x_set_ids, x_hat_ids, y_set, y_hat):
        feature_set_batch = []
        feature_hat_batch = []

        # loop through all features
        for feature in self.data:
            f_set = np.array([np.array(feature[b]) for b in x_set_ids])
            f_hat = np.array(feature[x_hat_ids])
            #print(f_set.shape)
            #print(f_hat.shape)

            f_set = f_set.reshape((self.batch_size, 2, self.k_shot, *(feature.shape[1:])))

            feature_set_batch.append(f_set)
            feature_hat_batch.append(f_hat)

            return (feature_set_batch, y_set, feature_hat_batch, y_hat)

# data converter function
def data_converter(support_data, query_data_x, query_data_y, n_shot, predict_mode=False):
    """
    convert the data to the fomat of Protonet class, only apply to two classes

    Args:
      support_data(numpy.array): the support data, output of Data_loader::next_batch_gen()(x[0])
      query_data_x(numpy.array): the query data, output of Data_loader::next_batch_gen()(x[1])
      query_data_y(numpy.array): the query data label, output of Data_loader::next_batch_gen(y[0])

    Returns:
      x_support(torch.Tensor): the support shape: (n_class, n_support, *data dimension)
      x_query(torch.Tensor): the query set, shape: (n_class, n_query, *data dimension), x_query::n_class == x_support::n_class
      y_query(torch.longTensor): the label for query set (n_class, n_query, 1), [positive, negative]
    """

    # support data
    x_support_pos = torch.Tensor(support_data[:, :n_shot, :, :].reshape(-1, *support_data.shape[2:]))
    x_support_neg = torch.Tensor(support_data[:, n_shot:, :, :].reshape(-1, *support_data.shape[2:]))
    x_support = torch.stack([x_support_pos, x_support_neg])

    # query data
    x_query = torch.Tensor(query_data_x.reshape(2, -1, *query_data_x.shape[1:]))

    # y_query
    if predict_mode is False:
        y_query = Variable(torch.LongTensor(query_data_y).view(2, -1, 1), requires_grad=False)
    else:
        y_query = None

    return x_support, x_query, y_query

# data corruption functions
def random_pop(arr, ratio):
    pop_size = ceil(len(arr) * ratio)
    pop_result = []
    arr = arr.copy().tolist()
    random.shuffle(arr)

    for _ in range(pop_size):
        pop_result.append(arr.pop())

    return np.array(pop_result), np.array(arr)

def corruption_func(pos_positions, neg_positions, corruption_ratio=0.5):
    # copy data
    pos_positions = pos_positions.copy()
    neg_positions = neg_positions.copy()

    # random choose a direction
    direction = random.choice(['pos', 'neg'])

    # corruption:
    if direction == 'pos':
        temp, neg_positions = random_pop(neg_positions, corruption_ratio)
        pos_positions = np.concatenate((pos_positions, temp), axis=0)
    else:
        temp, pos_positions = random_pop(pos_positions, corruption_ratio)
        neg_positions = np.concatenate((neg_positions, temp), axis=0)

    return pos_positions, neg_positions, direction


# model
# base CNN model
class Base_CNN(nn.Module):
    def __init__(self, emb_dim, num_filter, kernel_sizes):
        # initialization
        super(Base_CNN, self).__init__()
        self.emb_dim = emb_dim
        self.num_filter = num_filter
        self.kernel_sizes = kernel_sizes
        # convolution layers
        self.convs = nn.ModuleList([nn.Conv2d(1, self.num_filter, (f, self.emb_dim)) for f in self.kernel_sizes])

    def forward(self, x):
        x = x.unsqueeze(1)
        x = [F.relu(conv(x).squeeze(-1)) for conv in self.convs]  # output of three conv
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # continue with 3 maxpooling
        x = torch.cat(x, 1)

        return x

# prototypical wrapper
class Protonet(nn.Module):
    def __init__(self, encoder):
        super(Protonet, self).__init__()
        self.encoder = encoder

    @staticmethod
    def euclidean_dist(x, y):
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)

    def forward(self, x_support, x_query, y_query):
        """
        Forward for prototypical network

        Args:
          x_support(torch.Tensor): the support set, assume the shape (n_class, n_support, *data dimension)
          x_query(torch.Tensor): the query set, assume the shape (n_class, n_query, *data dimension), x_query::n_class == x_support::n_class
          y_query(torch.longTensor): the label for query set (n_class, n_query, 1)

        returns:
          loss(torch.Tensor): negative log likelihood to be minimized, will be used to update the parameters via backprobagation
          acc(float): accuracy
        """

        # find number of class, number of query, number of support
        n_class = x_support.size(0)
        n_support = x_support.size(1)
        n_query = x_query.size(1)

        # concat the support and query to pass the encoder at one time
        x = torch.cat([x_support.view(n_class * n_support, *x_support.size()[2:]), # shape (n_class * n_support, *data dimension)
                       x_query.view(n_class * n_query, *x_query.size()[2:])],  # shape: (n_class * n_support, *data dimension)
                      dim=0)
        z = self.encoder.forward(x)  # pass encoder
        z_dim = z.size(-1)  # dimension of latent vector

        # estimate the prototypes
        prototypes = z[:n_class * n_support].view(n_class, n_support, z_dim).mean(dim=1)  # take average on the different support vector(dim=1)

        # extract the latent query vector
        querys = z[n_class * n_support:]

        # calculate the distance
        dists = self.euclidean_dist(querys, prototypes)

        # loss
        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)  # log probability
        loss = -log_p_y.gather(2, y_query).squeeze().view(-1).mean()  # loss
        _, y_hat = log_p_y.max(2) # return: max, max_indicies
        acc = torch.eq(y_hat, y_query.squeeze()).float().mean().item()

        return loss, acc

    def evaluation(self, x_support, x_query, y_query):
        """
        evaluation function, will return classes, accuracy, loss(float, requires_grad=False)

        Args:
          x_support(torch.Tensor): the support set, assume the shape (n_class, n_support, *data dimension)
          x_query(torch.Tensor): the query set, assume the shape (n_class, n_query, *data dimension), x_query::n_class == x_support::n_class
          y_query(torch.longTensor): the label for query set (n_class, n_query, 1)

        returns:
          y_hat(torch.Tensor): the prediction label
          loss(float): loss
          acc(float): accuracy
        """

        # find number of class, number of query, number of support
        n_class = x_support.size(0)
        n_support = x_support.size(1)
        n_query = x_query.size(1)

        # concat the support and query to pass the encoder at one time
        x = torch.cat([x_support.view(n_class * n_support, *x_support.size()[2:]), # shape (n_class * n_support, *data dimension)
                       x_query.view(n_class * n_query, *x_query.size()[2:])],  # shape: (n_class * n_support, *data dimension)
                      dim=0)
        z = self.encoder.forward(x)  # pass encoder
        z_dim = z.size(-1)  # dimension of latent vector

        # estimate the prototypes
        prototypes = z[:n_class * n_support].view(n_class, n_support, z_dim).mean(dim=1)  # take average on the different support vector(dim=1)

        # extract the latent query vector
        querys = z[n_class * n_support:]

        # calculate the distance
        dists = self.euclidean_dist(querys, prototypes)

        # loss
        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)  # log probability
        loss = -log_p_y.gather(2, y_query).squeeze().view(-1).mean().item()  # loss
        _, y_hat = log_p_y.max(2) # return: max, max_indicies
        acc = torch.eq(y_hat, y_query.squeeze()).float().mean().item()

        return y_hat, acc, loss

    def predict_prob(self, x_support, x_query):
        """

        make a prediction, return positive probability

        Args:
          x_support(torch.Tensor): the support set, assume the shape (n_class, n_support, *data dimension)
          x_query(torch.Tensor): the query set, assume the shape (n_class, n_query, *data dimension), x_query::n_class == x_support::n_class

        returns:
          p_y(float): the average probability
        """

        # find number of class, number of query, number of support
        n_class = x_support.size(0)
        n_support = x_support.size(1)
        n_query = x_query.size(1)

        # concat the support and query to pass the encoder at one time
        x = torch.cat([x_support.view(n_class * n_support, *x_support.size()[2:]), # shape (n_class * n_support, *data dimension)
                       x_query.view(n_class * n_query, *x_query.size()[2:])],  # shape: (n_class * n_support, *data dimension)
                      dim=0)
        z = self.encoder.forward(x)  # pass encoder
        z_dim = z.size(-1)  # dimension of latent vector

        # estimate the prototypes
        prototypes = z[:n_class * n_support].view(n_class, n_support, z_dim).mean(dim=1)  # take average on the different support vector(dim=1)

        # extract the latent query vector
        querys = z[n_class * n_support:]

        # calculate the distance
        dists = self.euclidean_dist(querys, prototypes)

        # probability
        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)  # log probability
        p_y = torch.exp(log_p_y).view(-1, 2).mean(dim=0) # average probability

        return p_y[0].item()


# train crowd functions
# train function
def train_proto(the_model,
                n_epochs,
                n_eposides_train,
                n_eposides_valid,
                learning_rate,
                train_loader,
                valid_loader,
                n_shot,
                the_device,
                batch_size=32,
                learning_rate_schedule=False,
                update_step=100,
                gamma=0.5,
                save_model=False,
                save_path=None,
                verbose=True):

    # move model
    the_model = the_model.to(the_device)

    # history
    history = {'train_acc': [], 'train_loss': [],
               'train_acc_avg': [], 'train_loss_avg': [],
               'valid_acc': [], 'valid_loss': [],
               'valid_acc_avg': [], 'valid_loss_avg': []}

    # set up optimizer
    optimizer = torch.optim.Adam(the_model.parameters(), lr=learning_rate)
    # if schedule learning rate
    if learning_rate_schedule is True:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                    step_size=update_step,
                                                    gamma=gamma)
        update_count = 0
        if verbose:
            print(f"Initial learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")

    for i in range(n_epochs):
        # train metrics
        train_cumulative_acc = 0
        train_cumulative_loss = 0
        for j in range(n_eposides_train):
            the_model.train()
            # retrieve data from train loader
            cur_support_data, cur_query_data_x, cur_query_data_y = train_loader.next_batch()
            cur_support_data = cur_support_data[0].numpy()
            cur_support_data = cur_support_data.reshape(batch_size, -1, 100, 768)  # hard-coded sentence embedding shape
            cur_query_data_x = cur_query_data_x[0].numpy()
            cur_query_data_y = cur_query_data_y.numpy()
            # transform data
            cur_x_support, cur_x_query, cur_y_query = data_converter(support_data=cur_support_data,
                                                                     query_data_x=cur_query_data_x,
                                                                     query_data_y=cur_query_data_y,
                                                                     n_shot=n_shot)
            # move data to gpu:
            cur_x_support = cur_x_support.to(device)
            cur_x_query = cur_x_query.to(device)
            cur_y_query = cur_y_query.to(device)
            # forward
            cur_loss, cur_acc = the_model(cur_x_support, cur_x_query, cur_y_query)
            # backward
            cur_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # learning rate schedule
            if learning_rate_schedule is True:
                scheduler.step()
                update_count += 1
                if update_count % update_step == 0 and verbose:
                    print(f"Update learning rate to {optimizer.state_dict()['param_groups'][0]['lr']}")
            # cumulative loss and acc
            train_cumulative_acc += cur_acc
            train_cumulative_loss += cur_loss.item()
            # record
            history['train_acc'].append(cur_acc)
            history['train_loss'].append(cur_loss.item())

        # evaluation metrics
        valid_cumulative_acc = 0
        valid_cumulative_loss = 0
        valid_pred = []
        valid_label = []
        for k in range(n_eposides_valid):
            the_model.eval()
            # retrieve data from validation loader
            cur_support_data, cur_query_data_x, cur_query_data_y = next(valid_loader)
            cur_support_data = cur_support_data[0].numpy()
            cur_support_data = cur_support_data.reshape(batch_size, -1, 100, 768)  # hard-coded sentence embedding shape
            cur_query_data_x = cur_query_data_x[0].numpy()
            cur_query_data_y = cur_query_data_y.numpy()
            # transform data
            cur_x_support, cur_x_query, cur_y_query = data_converter(support_data=cur_support_data,
                                                                     query_data_x=cur_query_data_x,
                                                                     query_data_y=cur_query_data_y,
                                                                     n_shot=n_shot)
            # move data to gpu:
            cur_x_support = cur_x_support.to(device)
            cur_x_query = cur_x_query.to(device)
            cur_y_query = cur_y_query.to(device)
            # evaluation
            y_hat, acc, loss = the_model.evaluation(cur_x_support, cur_x_query, cur_y_query)
            # metrics
            valid_pred.append(y_hat.long().cpu().numpy())
            valid_label.append(cur_query_data_y)
            valid_cumulative_acc += acc
            valid_cumulative_loss += loss
            # record
            history['valid_acc'].append(acc)
            history['valid_loss'].append(loss)
        valid_pred = np.concatenate(valid_pred).reshape(-1)
        valid_label = np.concatenate(valid_label).reshape(-1)


        # record
        history['train_acc_avg'].append(train_cumulative_acc / n_eposides_train)
        history['train_loss_avg'].append(train_cumulative_loss / n_eposides_train)
        history['valid_acc_avg'].append(valid_cumulative_acc / n_eposides_valid)
        history['valid_loss_avg'].append(valid_cumulative_loss / n_eposides_valid)

        # verbose
        if verbose:
            print('=' * 10 + f'Epoch: {i + 1} / {n_epochs}' + '=' * 10)
            print(f'\nTrain acc: {train_cumulative_acc / n_eposides_train}, Train loss: {train_cumulative_loss / n_eposides_train}')
            print(f'\nValidation acc: {valid_cumulative_acc / n_eposides_valid}, Validation loss: {valid_cumulative_loss / n_eposides_valid}')
            print('\n')
            # print(classification_report(y_true=valid_label, y_pred=valid_pred))
            print('=' * 37)

    # save model
    if save_model:
        torch.save(the_model.state_dict(), save_path)

    return history

# single crowd train function
def train_single_crowd_member(data,
                              aug_data,
                              X_train_pos,
                              X_train_neg,
                              X_val_pos,
                              X_val_neg,
                              model_save_path,
                              device,
                              batch_size=32,
                              k_shot=5,
                              training_eposides=10,
                              validation_eposides=10,
                              learning_rate=0.0001,
                              n_epochs=10,
                              corruption_ratio=0.5,
                              num_filter=20):
    # model set up
    cnn_model = Base_CNN(emb_dim=768, num_filter=num_filter, kernel_sizes=[1, 3, 5])
    proto_model = Protonet(cnn_model)

    # data set up
    data = [data]
    aug_data = [aug_data]

    # subsample & corruption
    X_train_pos, X_train_neg, cur_direction = corruption_func(pos_positions=X_train_pos,
                                                                neg_positions=X_train_neg,
                                                                corruption_ratio=corruption_ratio)

    # data loader set up
    train_loader = CL_Data_loader(X_train_pos=X_train_pos,
                                  X_train_neg=X_train_neg,
                                  X_val_pos=X_val_pos,
                                  X_val_neg=X_val_neg,
                                  data=data,
                                  aug_data=aug_data,
                                  batch_size=batch_size,
                                  k_shot=k_shot,
                                  train_mode=True)
    # training_generator = train_loader.next_batch_gen()

    # eval generator
    eval_loader = CL_Data_loader(X_train_pos=X_train_pos,
                                 X_train_neg=X_train_neg,
                                 X_val_pos=X_val_pos,
                                 X_val_neg=X_val_neg,
                                 data = data,
                                 aug_data=aug_data,
                                 batch_size = batch_size,
                                 k_shot = k_shot,
                                 train_mode = False)
    validation_generator = eval_loader.next_eval_batch_gen()

    # train model
    the_history = train_proto(the_model=proto_model,
                              n_epochs=n_epochs,
                              n_eposides_train=training_eposides,
                              n_eposides_valid=validation_eposides,
                              learning_rate=learning_rate,
                              train_loader=train_loader,
                              valid_loader=validation_generator,
                              n_shot=k_shot,
                              the_device=device,
                              batch_size=batch_size,
                              save_model=True,
                              save_path=model_save_path,
                              verbose=False)

    return num_filter, cur_direction, corruption_ratio

# multiple crowd train function
def train_crowd(data,
                aug_data,
                X_train_pos,
                X_train_neg,
                X_val_pos,
                X_val_neg,
                model_save_folder,
                device,
                corruption_ratios,
                num_filters,
                batch_size=32,
                k_shot=5,
                training_eposides=10,
                validation_eposides=10,
                learning_rate=0.0001,
                n_epochs=10):

    # combinations
    combinations = [(x, y) for x in corruption_ratios for y in num_filters]

    # record
    crowd_id_record = []
    filter_record = []
    direction_record = []
    corruption_ratio_record = []

    for i, temp in enumerate(combinations):
        cur_corruption_ratio, cur_num_filter = temp
        print(f'Training {i} member')

        cur_save_path = os.path.join(model_save_folder, str(i) + '.pth')

        _, cur_direction, _ = train_single_crowd_member(data=data,
                                  aug_data=aug_data,
                                  X_train_pos=X_train_pos,
                                  X_train_neg=X_train_neg,
                                  X_val_pos=X_val_pos,
                                  X_val_neg=X_val_neg,
                                  model_save_path=cur_save_path,
                                  device=device,
                                  batch_size=batch_size,
                                  k_shot=k_shot,
                                  training_eposides=training_eposides,
                                  validation_eposides=validation_eposides,
                                  learning_rate=learning_rate,
                                  n_epochs=n_epochs,
                                  corruption_ratio=cur_corruption_ratio,
                                  num_filter=cur_num_filter)

        # record
        crowd_id_record.append(i)
        filter_record.append(cur_num_filter)
        direction_record.append(cur_direction)
        corruption_ratio_record.append(cur_corruption_ratio)

    # save to csv
    pd.DataFrame({'CrowdID': crowd_id_record, 'NumFilters': filter_record, 'Direction': direction_record, 'CorruptionRatio': corruption_ratio_record}).to_csv(os.path.join(model_save_folder, 'crowd_info.csv'))


# train crowd job
def train_crowd_job():
    # verbosity
    print('Crowd Training Started')
    # parameters setting
    # class_names = ['Study Period', 'Perspective', 'Population', 'Sample Size', 'Intervention', 'Country']
    class_names = ['Country']  # FIXME
    batch_size = 32
    k_shot = 5
    training_eposides = 10
    validation_eposides = 10
    learning_rate = 0.0001
    n_epochs = 10
    # filters = list(np.arange(5, 40, 5))
    # corruption_ratios = list(np.arange(0.05, 0.6, 0.05))
    filters = list(np.arange(5, 40, 30))  # FIXME
    corruption_ratios = list(np.arange(0.05, 0.6, 0.6))  # FIXME


    for class_name in class_names:
        print(f'Class: {class_name} started')
        # crowd path
        if os.path.isdir(os.path.join(cwd, 'crowd', class_name)):
            crowd_path = os.path.join(cwd, 'crowd', class_name)
        else:
            os.mkdir(os.path.join(cwd, 'crowd', class_name))
            crowd_path = os.path.join(cwd, 'crowd', class_name)

        # data
        data = token_embeddings
        aug_data = aug_token_embeddings
        X_train_pos=train_pos_positions[class_name]
        X_train_neg=train_neg_positions[class_name]
        X_val_pos=valid_pos_positions[class_name]
        X_val_neg=valid_neg_positions[class_name]


        # train
        train_crowd(data=data,
                    aug_data=aug_data,
                    X_train_pos=X_train_pos,
                    X_train_neg=X_train_neg,
                    X_val_pos=X_val_pos,
                    X_val_neg=X_val_neg,
                    model_save_folder=crowd_path,
                    device=device,
                    corruption_ratios=corruption_ratios,
                    num_filters=filters,
                    batch_size=batch_size,
                    k_shot=k_shot,
                    training_eposides=training_eposides,
                    validation_eposides=validation_eposides,
                    learning_rate=learning_rate,
                    n_epochs=n_epochs)
        print('Crowd Training Finished')
        print('-' * 30)


# Evaluation functions
# location finder
class LocationFinder:
    def __init__(self, embedding_data):
        self.embedding_data = embedding_data
    
    def find_one(self, vec):
        for i in range(self.embedding_data.shape[0]):
            cur_array = self.embedding_data[i]
            if np.all(cur_array == vec):
                return i
    
    def find_locations(self, vec):
        result = []
        for i in range(vec.shape[0]):
            cur_array = vec[i]
            result.append(self.find_one(cur_array))
        
        return result

# sample data class
class DataSampler:
    def __init__(self, embedding_data, aug_data, X_train_pos, X_train_neg, X_val_pos, X_val_neg):
        self.embedding_data = [embedding_data]
        self.aug_data = [aug_data]
        self.X_train_pos = X_train_pos
        self.X_train_neg = X_train_neg
        self.X_val_pos = X_val_pos
        self.X_val_neg = X_val_neg

    
    def sample_one_batch(self, col_name, sample_size, k_shot=5):
        ret_support_data = []
        ret_query_data_x = []
        ret_query_data_y = []
        ret_alpha = []
        ret_aug_type = []
        ret_support_ids = []
        ret_query_ids = []

        # construct data loader & generator
        loader = CL_Data_loader(X_train_pos=self.X_train_pos[col_name], 
                            X_train_neg=self.X_train_neg[col_name],
                            X_val_pos=self.X_val_pos[col_name], 
                            X_val_neg=self.X_val_neg[col_name],
                            data=self.embedding_data,
                            aug_data=self.aug_data,
                            batch_size=1,
                            k_shot=k_shot,
                            train_mode = True)
        for _ in range(sample_size):
            # random sample alpha & aug_type
            cur_alpha = random.randint(0,5)
            cur_aug_type = random.randint(1, 4)
            ret_alpha.append(cur_alpha)
            ret_aug_type.append(cur_aug_type)
            # sample data
            cur_support_data, cur_query_data_x, cur_query_data_y, cur_support_ids, cur_query_ids = loader.next_batch(alpha=cur_alpha, aug_type=cur_aug_type, return_sample_ids=True)  # FIXME: Get id from .next_batch() function
            cur_support_data = cur_support_data[0].numpy()
            cur_support_data = cur_support_data.reshape(1, -1, 100, 768)  # hard-coded sentence embedding shape
            cur_query_data_x = cur_query_data_x[0].numpy()
            cur_query_data_y = cur_query_data_y.numpy()
            # append
            ret_support_data.append(cur_support_data)
            ret_query_data_x.append(cur_query_data_x)
            ret_query_data_y.append(cur_query_data_y)
            ret_support_ids.append(cur_support_ids[0])
            ret_query_ids.append(cur_query_ids[0])
        
        # concat
        ret_support_data = np.concatenate(ret_support_data)
        ret_query_data_x = np.concatenate(ret_query_data_x)
        ret_query_data_y = np.concatenate(ret_query_data_y)

        return ret_support_data, ret_query_data_x, ret_query_data_y, ret_alpha, ret_aug_type, ret_support_ids, ret_query_ids

    def sample_one_batch_valid(self, col_name, sample_size, k_shot=5):
        ret_support_data = []
        ret_query_data_x = []
        ret_query_data_y = []
        ret_alpha = []
        ret_aug_type = []
        ret_support_ids = []
        ret_query_ids = []

        # construct data loader & generator
        loader = CL_Data_loader(X_train_pos=self.X_train_pos[col_name],
                                X_train_neg=self.X_train_neg[col_name],
                                X_val_pos=self.X_val_pos[col_name],
                                X_val_neg=self.X_val_neg[col_name],
                                data=self.embedding_data,
                                aug_data=self.aug_data,
                                batch_size=1,
                                k_shot=k_shot,
                                train_mode = False)
        loader_gen = loader.next_eval_batch_gen(return_sample_ids=True)
        for _ in range(sample_size):
            # random sample alpha & aug_type
            cur_alpha = 0
            cur_aug_type = 0
            ret_alpha.append(cur_alpha)
            ret_aug_type.append(cur_aug_type)
            # sample data
            cur_support_data, cur_query_data_x, cur_query_data_y, cur_support_ids, cur_query_ids = next(loader_gen)
            cur_support_data = cur_support_data[0].numpy()
            cur_support_data = cur_support_data.reshape(1, -1, 100, 768)  # hard-coded sentence embedding shape
            cur_query_data_x = cur_query_data_x[0].numpy()
            cur_query_data_y = cur_query_data_y.numpy()
            # append
            ret_support_data.append(cur_support_data)
            ret_query_data_x.append(cur_query_data_x)
            ret_query_data_y.append(cur_query_data_y)
            ret_support_ids.append(cur_support_ids[0])
            ret_query_ids.append(cur_query_ids[0])

        # concat
        ret_support_data = np.concatenate(ret_support_data)
        ret_query_data_x = np.concatenate(ret_query_data_x)
        ret_query_data_y = np.concatenate(ret_query_data_y)

        return ret_support_data, ret_query_data_x, ret_query_data_y, ret_alpha, ret_aug_type, ret_support_ids, ret_query_ids

# split small
class split_small(Dataset):
    def __init__(self, cur_support_data, cur_query_data_x, cur_query_data_y):
        self.cur_support_data = cur_support_data
        self.cur_query_data_x = cur_query_data_x
        self.cur_query_data_y = cur_query_data_y
    
    def __getitem__(self, index):
        return self.cur_support_data[index], self.cur_query_data_x[index], self.cur_query_data_y[index]
    
    def __len__(self):
        return self.cur_query_data_y.shape[0]

# given data evaluation
def given_data_evaluation(cur_support_data, cur_query_data_x, cur_query_data_y, model, k_shot=5):
    # result
    judgemet = []
    temp_dataset = split_small(cur_support_data, cur_query_data_x, cur_query_data_y)
    temp_dataloader = DataLoader(temp_dataset, batch_size=128, shuffle=False)
    
    for cur_x_support_temp, cur_x_query_temp, cur_y_query_temp in temp_dataloader:
        x_support, x_query, y_query = data_converter(support_data=cur_x_support_temp.numpy(),
                                                              query_data_x=cur_x_query_temp.numpy(),
                                                              query_data_y=cur_y_query_temp.numpy(),
                                                              n_shot=k_shot)
        # move to device
        x_support = x_support.to(device)
        x_query = x_query.to(device)
        y_query = y_query.to(device)
        # predict label
        yhat, _, _ = model.evaluation(x_support, x_query, y_query)
        # reshape
        yhat = yhat.long().cpu().numpy().reshape(-1)
        y = y_query.long().cpu().numpy().reshape(-1)
        # check if prediction is correct
        cur_judgemet = np.where(yhat == y, 1, 0)
        judgemet.append(cur_judgemet)

    return np.concatenate(judgemet)


# Evaluation jobs
# train evaluation job
def train_evaluation_job():
    print('Train Evaluation Job Started')
    # set up parameters
    # sample_question_size = 35_000
    sample_question_size = 10  # FIXME
    # col_names = ['Study Period', 'Perspective', 'Population', 'Sample Size', 'Intervention', 'Country']
    col_names = ['Country']  # FIXME
    # chunk_size = 5000
    chunk_size = 10  # FIXME
    num_chunks = ceil(sample_question_size / chunk_size)
    # set up path
    crowd_base_path = os.path.join(cwd, 'crowd')
    task_info_path = os.path.join(cwd, 'train_data', 'task')
    model_judgement_path = os.path.join(cwd, 'train_data', 'model_judgement')
    # initialize
    mySampler = DataSampler(embedding_data=token_embeddings, 
                            aug_data=aug_token_embeddings,
                            X_train_pos=train_pos_positions, 
                            X_train_neg=train_neg_positions, 
                            X_val_pos=valid_pos_positions, 
                            X_val_neg=valid_neg_positions)
    # chunk evaluation
    for cur_col in col_names:
        print(f'{cur_col} started\n')
        for j in tqdm(range(num_chunks)):
            # sample data
            cur_support_data, cur_query_data_x, cur_query_data_y, ret_alpha, ret_aug_type, ret_support_ids, ret_query_ids = mySampler.sample_one_batch(col_name=cur_col, sample_size=chunk_size)  # FIXME: call the sampler function
            # print('Sample Finished')
            # construct the task information
            cur_query_locs = ret_query_ids
            cur_support_pos_locs = []
            cur_support_neg_locs = []
            for i in range(len(ret_support_ids)):
                loc_temp = ret_support_ids[i]
                cur_support_pos_locs.append(loc_temp[5:])
                cur_support_neg_locs.append(loc_temp[:5])
            ids = [str(uuid.uuid4()) for _ in range(len(cur_query_locs))]
            cur_record = {'ID': ids,
                        'Pos_support_locs': cur_support_pos_locs,
                        'Neg_support_locs': cur_support_neg_locs,
                        'Query_loc': cur_query_locs,
                        'Label': cur_query_data_y,
                        'Alpha': ret_alpha,
                        'Aug_type': ret_aug_type}
            pd.DataFrame(cur_record).to_csv(os.path.join(task_info_path, 'chuncks', cur_col, f'c_{j}' +'.csv'), index=False)

            # models path
            models_path_long = list(glob.glob(os.path.join(crowd_base_path, 'Country', '*.pth')))
            models_path = [os.path.basename(f) for f in models_path_long]
            model_info_csv = list(glob.glob(os.path.join(crowd_base_path, 'Country', '*.csv')))
            model_info_csv = model_info_csv[0]
            model_info_csv = pd.read_csv(model_info_csv, index_col=0)
            filters_dict = {}
            for _, cur_row in model_info_csv.iterrows():
                filters_dict[cur_row['CrowdID']] = cur_row['NumFilters']
            for cur_path in tqdm(models_path):
                # print(f'Model {cur_path} started')
                # load model
                cnn_lstm = Base_CNN(emb_dim=768, num_filter=filters_dict[int(cur_path.split('.')[0])], kernel_sizes=[1, 3, 5])
                prto_model = Protonet(cnn_lstm).to(device)
                prto_model.load_state_dict(torch.load(os.path.join(crowd_base_path, cur_col, cur_path)))
                prto_model.eval()
                # evaluation
                judgement = given_data_evaluation(cur_support_data, cur_query_data_x, cur_query_data_y, prto_model)
                if not os.path.isdir(os.path.join(model_judgement_path, cur_col)):
                    os.mkdir(os.path.join(model_judgement_path, cur_col))
                pd.DataFrame({'ID': ids, 'Judgement': judgement}).to_csv(os.path.join(model_judgement_path, cur_col, 'chunk', cur_path + f'_c_{j}' + '.csv'), index=False)
    
    # merge chunk
    for cur_col in col_names:
        # combine task
        temp_list = []
        file_chunks = os.listdir(os.path.join(task_info_path, 'chuncks', cur_col))
        for cur_file in tqdm(file_chunks):
            temp_list.append(pd.read_csv(os.path.join(task_info_path, 'chuncks', cur_col, cur_file)))
        temp_combined = pd.concat(temp_list, axis=0, ignore_index=True)
        temp_combined = temp_combined.sort_values(by='ID').reset_index()
        del temp_combined['index']
        # check duplicates
        duplicates_check_df = temp_combined.loc[:, temp_combined.columns != 'ID'].copy()
        duplicates_boolean = duplicates_check_df.duplicated(subset=None, keep='first').tolist()
        print(f'Class :{cur_col}, num of duplicates: {np.sum(duplicates_boolean)}')
        duplicates_boolean = [not item for item in duplicates_boolean]  # flip the boolean to index dataframe
        # remove duplicates from combined and save
        temp_combined = temp_combined[duplicates_boolean].reset_index()
        del temp_combined['index']
        temp_combined.to_csv(os.path.join(task_info_path, cur_col + '.csv'), index=False)
        # combine model judgement
        model_judgement_chunk_path = os.path.join(model_judgement_path, cur_col, 'chunk')
        models_path = os.listdir(os.path.join(crowd_base_path, cur_col))
        for cur_path in models_path:
            # combine
            temp_list = []
            for cur_file in glob.glob(os.path.join(model_judgement_path, cur_col, 'chunk', cur_path + '_c_?.csv')):
                temp_list.append(pd.read_csv(cur_file))
            temp_combined = pd.concat(temp_list, axis=0, ignore_index=True)
            temp_combined = temp_combined.sort_values(by='ID').reset_index()
            del temp_combined['index']
            # remove duplicated lines
            temp_combined = temp_combined[duplicates_boolean].reset_index()
            del temp_combined['index']
            # save
            temp_combined.to_csv(os.path.join(os.path.join(model_judgement_path, cur_col, cur_path + '.csv')), index=False)
    print("Train Evaluation Finished")
    print('-' * 30)

# valid evaluation job
def valid_evaluation_job():
    print('Valid Evaluation Started')
    # set up parameters
    # sample_question_size = 15_000
    sample_question_size = 10 # FIXME
    # col_names = ['Study Period', 'Perspective', 'Population', 'Sample Size', 'Intervention', 'Country']
    col_names = ['Country'] # FIXME
    # chunk_size = 5000
    chunk_size = 10  # FIXME
    num_chunks = ceil(sample_question_size / chunk_size)
    # set up path
    crowd_base_path = os.path.join(cwd, 'crowd')
    task_info_path = os.path.join(cwd, 'valid_data', 'task')
    model_judgement_path = os.path.join(cwd, 'valid_data', 'model_judgement')
    # initialize
    mySampler = DataSampler(embedding_data=token_embeddings,
                            aug_data=aug_token_embeddings,
                            X_train_pos=train_pos_positions,
                            X_train_neg=train_neg_positions,
                            X_val_pos=valid_pos_positions,
                            X_val_neg=valid_neg_positions)
    # chunk evaluation
    # evaluation
    for cur_col in col_names:
        print(f'{cur_col} started\n')
        for j in tqdm(range(num_chunks)):
            # sample data
            cur_support_data, cur_query_data_x, cur_query_data_y, ret_alpha, ret_aug_type, ret_support_ids, ret_query_ids = mySampler.sample_one_batch_valid(col_name=cur_col, sample_size=chunk_size)  # FIXME: call the sampler function
            # print('Sample Finished')
            # construct the task information
            cur_query_locs = ret_query_ids
            cur_support_pos_locs = []
            cur_support_neg_locs = []
            for i in range(len(ret_support_ids)):
                loc_temp = ret_support_ids[i]
                cur_support_pos_locs.append(loc_temp[5:])  # FIXME: Positive Label later
                cur_support_neg_locs.append(loc_temp[:5])  # FIXME: Negative label first
            ids = [str(uuid.uuid4()) for _ in range(len(cur_query_locs))]
            cur_record = {'ID': ids,
                        'Pos_support_locs': cur_support_pos_locs,
                        'Neg_support_locs': cur_support_neg_locs,
                        'Query_loc': cur_query_locs,
                        'Label': cur_query_data_y,
                        'Alpha': ret_alpha,
                        'Aug_type': ret_aug_type}
            pd.DataFrame(cur_record).to_csv(os.path.join(task_info_path, 'chuncks', cur_col, f'c_{j}' +'.csv'), index=False)

            # models path
            models_path_long = list(glob.glob(os.path.join(crowd_base_path, 'Country', '*.pth')))
            models_path = [os.path.basename(f) for f in models_path_long]
            model_info_csv = list(glob.glob(os.path.join(crowd_base_path, 'Country', '*.csv')))
            model_info_csv = model_info_csv[0]
            model_info_csv = pd.read_csv(model_info_csv, index_col=0)
            filters_dict = {}
            for _, cur_row in model_info_csv.iterrows():
                filters_dict[cur_row['CrowdID']] = cur_row['NumFilters']
            for cur_path in tqdm(models_path):
                # print(f'Model {cur_path} started')
                # load model
                cnn_lstm = Base_CNN(emb_dim=768, num_filter=filters_dict[int(cur_path.split('.')[0])], kernel_sizes=[1, 3, 5])
                prto_model = Protonet(cnn_lstm).to(device)
                prto_model.load_state_dict(torch.load(os.path.join(crowd_base_path, cur_col, cur_path)))
                prto_model.eval()
                # evaluation
                judgement = given_data_evaluation(cur_support_data, cur_query_data_x, cur_query_data_y, prto_model)
                if not os.path.isdir(os.path.join(model_judgement_path, cur_col)):
                    os.mkdir(os.path.join(model_judgement_path, cur_col))
                pd.DataFrame({'ID': ids, 'Judgement': judgement}).to_csv(os.path.join(model_judgement_path, cur_col, 'chunk', cur_path + f'_c_{j}' + '.csv'), index=False)
    print('Valid Evaluation Finished')
    print('-' * 30)

# evaluation job
def evaluation_job():
    print("Evaluation Started")
    train_evaluation_job()
    valid_evaluation_job()
    print("Evaluation Finished")
    print('-' * 30)


# generate the jsonlines file
# train valid ID job
def train_valid_ID_job():
    print('Train Valid ID Job Started')
    # save train & valid task id
    # col_names = ['Study Period', 'Perspective', 'Population', 'Sample Size', 'Intervention', 'Country']
    col_names = ['Country']  # FIXME
    # train
    train_task_id = {}
    for cur_col in col_names:
        cur_task_df = pd.read_csv(os.path.join(train_path, 'task', cur_col + '.csv'))
        cur_task_ids = cur_task_df['ID'].tolist()
        train_task_id[cur_col] = cur_task_ids
    # save
    with open(os.path.join(total_path, 'train_task_id.pkl'), 'wb') as f:
        pickle.dump(train_task_id, f)
    # valid
    valid_task_id = {}
    for cur_col in col_names:
        cur_task_df = pd.read_csv(os.path.join(valid_path, 'task', cur_col + '.csv'))
        cur_task_ids = cur_task_df['ID'].tolist()
        valid_task_id[cur_col] = cur_task_ids
    # save
    with open(os.path.join(total_path, 'valid_task_id.pkl'), 'wb') as f:
        pickle.dump(valid_task_id, f)

    # sanity check
    for cur_col in col_names:
        print(cur_col)
        print(set(train_task_id[cur_col]).intersection(set(valid_task_id[cur_col])))
    print('Train Valid ID Job Finished')
    print('-' * 30)

def merging_job():
    print('Merging Started')
    # merge task
    # col_names = ['Study Period', 'Perspective', 'Population', 'Sample Size', 'Intervention', 'Country']
    col_names = ['Country']  # FIXME
    for cur_col in col_names:
        train_task_df = pd.read_csv(os.path.join(train_path, 'task', cur_col + '.csv'))
        train_task_df['Train'] = True
        valid_task_df = pd.read_csv(os.path.join(valid_path, 'task', cur_col + '.csv'))
        valid_task_df['Train'] = False
        total_task_df = pd.concat([train_task_df, valid_task_df], axis=0, ignore_index=True)
        total_task_df.to_csv(os.path.join(total_path, 'task', cur_col + '.csv'), index=False)

    # merge model_judgement
    for cur_col in col_names:
        print(cur_col, 'started')
        train_judgement_files = os.listdir(os.path.join(train_path, 'model_judgement', cur_col))
        train_judgement_files = [i for i in train_judgement_files if i != 'chunk']
        for cur_file in tqdm(train_judgement_files):
            cur_train_df = pd.read_csv(os.path.join(train_path, 'model_judgement', cur_col, cur_file))
            cur_valid_df = pd.read_csv(os.path.join(valid_path, 'model_judgement', cur_col, cur_file))
            cur_total_df = pd.concat([cur_train_df, cur_valid_df], axis=0, ignore_index=True)
            cur_total_df.to_csv(os.path.join(total_path, 'model_judgement', cur_col, cur_file), index=False)
    print('Merging Finished')
    print('-' * 30)

def format_jsonlines_job():
    print("FormatJsonlinesJob Started")
    # get file names
    total_model_judgement_path = os.path.join(total_path, 'model_judgement')
    folder_names = [cur_file for cur_file in os.listdir(total_model_judgement_path) if os.path.isdir(os.path.join(total_model_judgement_path, cur_file))]
    file_names = {}
    for cur_folder in folder_names:
        file_names[cur_folder] = glob.glob(os.path.join(total_model_judgement_path, cur_folder, '*.csv'))
    for cur_folder in folder_names:
        print(f'{cur_folder} started')
        cur_model_ids = [os.path.basename(cur_path).split('.')[0] for cur_path in file_names[cur_folder]]
        cur_dfs = []
        print('reading dfs')
        for cur_path in tqdm(file_names[cur_folder]):
            cur_dfs.append(pd.read_csv(cur_path))
        # cur_dfs = [pd.read_csv(cur_path) for cur_path in file_names[cur_folder]]
        cur_questions_ids = cur_dfs[0]['ID'].to_list()
        records = []
        print('generating .jsonlines')
        for i in tqdm(range(len(cur_model_ids))):
            temp_records = {}
            for cur_id in tqdm(cur_questions_ids):
                temp_records[cur_id] = cur_dfs[i][cur_dfs[i]['ID'] == cur_id]['Judgement'].to_list()[0]
            records.append({"subject_id": str(i), "responses": temp_records})
        with jsonlines.open(os.path.join(total_model_judgement_path, cur_folder + '.jsonlines'), 'w') as writer:
            writer.write_all(records)
    # get file names
    total_model_judgement_path = os.path.join(total_path, 'model_judgement')
    folder_names = [cur_file for cur_file in os.listdir(total_model_judgement_path) if os.path.isdir(os.path.join(total_model_judgement_path, cur_file))]
    file_names = {}
    for cur_folder in folder_names:
        file_names[cur_folder] = glob(os.path.join(total_model_judgement_path, cur_folder, '*.csv'))
    def job(cur_folder):
        print(f'{cur_folder} started')
        cur_model_ids = [os.path.basename(cur_path).split('.')[0] for cur_path in file_names[cur_folder]]
        cur_dfs = []
        # print('reading dfs')
        for cur_path in tqdm(file_names[cur_folder]):
            cur_dfs.append(pd.read_csv(cur_path))
        cur_questions_ids = cur_dfs[0]['ID'].to_list()
        records = []
        # print('generating .jsonlines')
        for i in tqdm(range(len(cur_model_ids))):
            temp_records = {}
            for cur_id in tqdm(cur_questions_ids):
                temp_records[cur_id] = cur_dfs[i][cur_dfs[i]['ID'] == cur_id]['Judgement'].to_list()[0]
            records.append({"subject_id": str(i), "responses": temp_records})
        with jsonlines.open(os.path.join(total_model_judgement_path, cur_folder + '.jsonlines'), 'w') as writer:
            writer.write_all(records)
        print(cur_folder, 'finished')
    pool = Pool(cpu_count() - 1)
    pool.map(job, folder_names)
    print("Format Jasonlines Job Finished")
    print('-' * 30)

def generate_jasonlines_job():
    print("Start generating jasonlines")
    train_valid_ID_job()
    merging_job()
    format_jsonlines_job()
    print('Jasonlines generated')
    print('-' * 30)

if __name__ == '__main__':
    # train_crowd_job()
    evaluation_job()
    generate_jasonlines_job()
