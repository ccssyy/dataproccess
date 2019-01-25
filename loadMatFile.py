import scipy.io as sio
import numpy as np
import h5py
import os


mat_file = 'alldata_v2.mat'

def load_mat(file_name):
    load_data = sio.loadmat(mat_file)
    # np.set_printoptions(precision=9)
    # print(load_data['ecg'])
    return load_data

def get_segment_average(mat_file):
    # 按段计算每段的平均血压值
    data = sio.loadmat(mat_file)
    average = []
    segment = 1
    segment_sum = 0
    count = 0
    for i in range(len(data['idx_segment'])):
        if data['idx_segment'][i][0] == segment:
            count += 1
            # a = data['nsbp'][i][0]
            segment_sum += int(data['nsbp'][i][0])
        else:
            average.append(segment_sum // count)
            count = 1
            segment = int(data['idx_segment'][i][0])
            segment_sum = int(data['nsbp'][i][0])
    average.append(segment_sum // count)

    segment_average = np.asarray(data['nsbp'], dtype=np.int32)
    # print(segment_average.shape)
    for i in range(len(segment_average)):
        idx = data['idx_segment'][i][0] - 1
        segment_average[i][0] = average[idx]
    return segment_average

def set_labels(data):
    label_dbp_c = []
    label_sbp_c = []
    print(len(data['ndbp']), len(data['nsbp']))
    for i in range(len(data['ndbp'])):
        if data['ndbp'][i] < 80:
            label_dbp_c.append(0)
        else:
            label_dbp_c.append(1)

        if data['nsbp'][i] < 130:
            label_sbp_c.append(0)
        else:
            label_sbp_c.append(1)

    print(len(label_sbp_c), len(label_dbp_c))
    return np.asarray(label_dbp_c, dtype=int), np.asarray(label_sbp_c, dtype=int)


def save2h5(data, output_file):
    if not os.path.exists(output_file):
        with h5py.File(output_file) as f:
            f['ppw1'] = data['ppw1']
            f['ppw2'] = data['ppw2']
            f['ecg'] = data['ecg']
            f['label_ndbp_r'] = data['ndbp']
            f['label_nsbp_r'] = data['nsbp']
            f['label_ndbp_c'], f['label_nsbp_c'] = set_labels(data)


def split_dataset(data, ratio=0.8):
    label_nsbp_c, label_ndbp_c = set_labels(data)
    data_size = len(data['ecg'])
    print(data_size)
    data_index = np.arange(data_size)
    np.random.shuffle(data_index)
    train_size = int(data_size * ratio)
    train_index = data_index[:train_size]
    test_index = data_index[train_size:]
    with h5py.File('train_set.h5') as train_file:
        with h5py.File('test_set.h5') as test_file:
            train_file['ecg'] = data['ecg'][train_index]
            test_file['ecg'] = data['ecg'][test_index]
            train_file['ppw1'] = data['ppw1'][train_index]
            test_file['ppw1'] = data['ppw1'][test_index]
            train_file['ppw2'] = data['ppw2'][train_index]
            test_file['ppw2'] = data['ppw2'][test_index]
            train_file['label_nsbp_r'] = data['nsbp'][train_index]
            test_file['label_nsbp_r'] = data['nsbp'][test_index]
            train_file['label_ndbp_r'] = data['ndbp'][train_index]
            test_file['label_ndbp_r'] = data['ndbp'][test_index]
            train_file['label_nsbp_c'] = label_nsbp_c[train_index]
            test_file['label_nsbp_c'] = label_nsbp_c[test_index]
            train_file['label_ndbp_c'] = label_ndbp_c[train_index]
            test_file['label_ndbp_c'] = label_ndbp_c[test_index]

def split_set_v2(data, ratio=0.8):
    label_nsbp_c, label_ndbp_c = set_labels(data)
    num = data['idx_subject'][-1]
    subject_ids = np.arange(1, num+1)
    test_ids = np.random.choice(subject_ids, size=int(num*(1-ratio)), replace=False)
    # print(test_ids)
    trainset_idx = []
    testset_idx = []
    for idx, subject in enumerate(data['idx_subject']):
        if subject in test_ids:
            testset_idx.append(idx)
        else:
            trainset_idx.append(idx)
    # print('trainset_idx:{}'.format(trainset_idx))
    # print('testset_idx:{}'.format(testset_idx))
    with h5py.File('train_set_v2.h5') as train_file:
        with h5py.File('test_set_v2.h5') as test_file:
            train_file['ecg'] = data['ecg'][trainset_idx]
            test_file['ecg'] = data['ecg'][testset_idx]
            train_file['ppw1'] = data['ppw1'][trainset_idx]
            test_file['ppw1'] = data['ppw1'][testset_idx]
            train_file['ppw2'] = data['ppw2'][trainset_idx]
            test_file['ppw2'] = data['ppw2'][testset_idx]
            train_file['label_nsbp_r'] = data['nsbp'][trainset_idx]
            test_file['label_nsbp_r'] = data['nsbp'][testset_idx]
            train_file['label_ndbp_r'] = data['ndbp'][trainset_idx]
            test_file['label_ndbp_r'] = data['ndbp'][testset_idx]
            train_file['label_nsbp_c'] = label_nsbp_c[trainset_idx]
            test_file['label_nsbp_c'] = label_nsbp_c[testset_idx]
            train_file['label_ndbp_c'] = label_ndbp_c[trainset_idx]
            test_file['label_ndbp_c'] = label_ndbp_c[testset_idx]

def split_set_v5(data, ratio=0.8):
    label_nsbp_c, label_ndbp_c = set_labels(data)
    num = data['idx_subject'][-1]
    subject_ids = np.arange(1, num+1)
    test_ids = np.random.choice(subject_ids, size=int(num*(1-ratio)), replace=False)
    # print(test_ids)
    trainset_idx = []
    testset_idx = []
    for idx, subject in enumerate(data['idx_subject']):
        if subject in test_ids:
            testset_idx.append(idx)
        else:
            trainset_idx.append(idx)
    # print('trainset_idx:{}'.format(trainset_idx))
    # print('testset_idx:{}'.format(testset_idx))

    with h5py.File('train_set_v6.h5') as train_file:
        with h5py.File('test_set_v6.h5') as test_file:
            train_file['ecg'] = data['ecg'][trainset_idx]
            test_file['ecg'] = data['ecg'][testset_idx]
            train_file['ppw1'] = data['ppw1'][trainset_idx]
            test_file['ppw1'] = data['ppw1'][testset_idx]
            train_file['ppw2'] = data['ppw2'][trainset_idx]
            test_file['ppw2'] = data['ppw2'][testset_idx]
            train_file['label_nsbp_r'] = data['nsbp'][trainset_idx]
            test_file['label_nsbp_r'] = data['nsbp'][testset_idx]
            train_file['label_ndbp_r'] = data['ndbp'][trainset_idx]
            test_file['label_ndbp_r'] = data['ndbp'][testset_idx]
            train_file['label_nsbp_c'] = label_nsbp_c[trainset_idx]
            test_file['label_nsbp_c'] = label_nsbp_c[testset_idx]
            train_file['label_ndbp_c'] = label_ndbp_c[trainset_idx]
            test_file['label_ndbp_c'] = label_ndbp_c[testset_idx]
            train_file['segment'] = data['idx_segment'][trainset_idx]
            test_file['segment'] = data['idx_segment'][testset_idx]
            train_file['segment_average'] = segment_average[trainset_idx]
            test_file['segment_average'] = segment_average[testset_idx]


def split_set_v4(data):
    label_nsbp_c, label_ndbp_c = set_labels(data)
    num = data['idx_subject'][-1]
    subject_ids = np.arange(1, num+1)
    # test_ids = np.random.choice(subject_ids, size=int(num*(1-ratio)), replace=False)
    # print(test_ids)
    trainset_idx = []
    testset_idx = []
    for idx, subject in enumerate(data['idx_dataset']):
        if subject == 1:
            trainset_idx.append(idx)
        else:
            testset_idx.append(idx)
    # print('trainset_idx:{}'.format(trainset_idx))
    # print('testset_idx:{}'.format(testset_idx))
    with h5py.File('train_set_v4.h5') as train_file:
        with h5py.File('test_set_v4.h5') as test_file:
            train_file['ecg'] = data['ecg'][trainset_idx]
            test_file['ecg'] = data['ecg'][testset_idx]
            train_file['ppw1'] = data['ppw1'][trainset_idx]
            test_file['ppw1'] = data['ppw1'][testset_idx]
            train_file['ppw2'] = data['ppw2'][trainset_idx]
            test_file['ppw2'] = data['ppw2'][testset_idx]
            train_file['label_nsbp_r'] = data['nsbp'][trainset_idx]
            test_file['label_nsbp_r'] = data['nsbp'][testset_idx]
            train_file['label_ndbp_r'] = data['ndbp'][trainset_idx]
            test_file['label_ndbp_r'] = data['ndbp'][testset_idx]
            train_file['label_nsbp_c'] = label_nsbp_c[trainset_idx]
            test_file['label_nsbp_c'] = label_nsbp_c[testset_idx]
            train_file['label_ndbp_c'] = label_ndbp_c[trainset_idx]
            test_file['label_ndbp_c'] = label_ndbp_c[testset_idx]
            train_file['segment'] = data['idx_segment'][trainset_idx]
            test_file['segment'] = data['idx_segment'][testset_idx]

def get_train_and_test_idx(mat_file, ratio=0.8):
    data = load_mat(mat_file)
    num = data['idx_subject'][-1]
    subject_ids = np.arange(1, num + 1)
    test_ids = np.random.choice(subject_ids, size=int(num * (1 - ratio)), replace=False)
    # print(test_ids)
    trainset_idx = []
    testset_idx = []
    for idx, subject in enumerate(data['idx_subject']):
        if subject in test_ids:
            testset_idx.append(idx)
        else:
            trainset_idx.append(idx)
    # print('trainset_idx:{}'.format(trainset_idx))
    # print('testset_idx:{}'.format(testset_idx))
    np.savetxt('trainset_idx.txt', trainset_idx, fmt='%d')
    np.savetxt('testset_idx.txt', testset_idx, fmt='%d')

def split_set_v6(data, segment_average, trainset_idx_file, testset_idx_file):
    label_nsbp_c, label_ndbp_c = set_labels(data)
    sbp = np.asarray(data['sbp'], dtype=np.int32)
    trainset_idx = np.loadtxt(trainset_idx_file, np.int32)
    # print('trainset_idx:{}'.format(trainset_idx))
    testset_idx = np.loadtxt(testset_idx_file, np.int32)
    average_sbp = []
    subject_sbp = np.zeros_like(data['idx_subject'], dtype=np.int32)    # 校准后所有数据条目的血压值
    segment_sbp_dict = {}     # 将血压值按段保存成字典 key为段号 value为该段号的血压值
    for i in range(len(sbp)):
        if data['idx_segment'][i][0] not in segment_sbp_dict.keys():
            segment_sbp_dict[data['idx_segment'][i][0]] = sbp[i]
    segment_sbp = np.asarray(list(segment_sbp_dict.values()))
    subject_id = 1
    segment_num = 0
    # subject_sum = 0
    for i in range(len(sbp)):
        if data['idx_subject'][i][0] == subject_id:
            continue
        else:
            rear_idx_segment = data['idx_segment'][i - 1][0]
            np.random.seed(100)
            average = np.mean(np.random.choice(segment_sbp[segment_num:rear_idx_segment].reshape(-1), 5, replace=False), dtype=np.int32)
            average_sbp.append(average)
            segment_num = data['idx_segment'][i-1][0]
            subject_id += 1
    average = np.mean(np.random.choice(segment_sbp[segment_num:data['idx_segment'][-1][0]].reshape(-1), 5, replace=False), dtype=np.int32)
    average_sbp.append(average)           # 每个人的基准校准血压
    print('average_sbp:{}'.format(average_sbp))
    #exit(0)
    for i in range(len(data['idx_subject'])):
        id_subject = data['idx_subject'][i][0]-1
        subject_sbp[i] = average_sbp[id_subject]
    print('subject_sbp:{}'.format(subject_sbp))
    # exit(0)

    with h5py.File('train_set_v8_random_5.h5') as train_file:
        with h5py.File('test_set_v8_random_5.h5') as test_file:
            train_file['ecg'] = data['ecg'][trainset_idx]
            test_file['ecg'] = data['ecg'][testset_idx]
            train_file['ppw1'] = data['ppw1'][trainset_idx]
            test_file['ppw1'] = data['ppw1'][testset_idx]
            train_file['ppw2'] = data['ppw2'][trainset_idx]
            test_file['ppw2'] = data['ppw2'][testset_idx]
            train_file['label_nsbp_r'] = data['nsbp'][trainset_idx]
            test_file['label_nsbp_r'] = data['nsbp'][testset_idx]
            train_file['label_ndbp_r'] = data['ndbp'][trainset_idx]
            test_file['label_ndbp_r'] = data['ndbp'][testset_idx]
            train_file['label_nsbp_c'] = label_nsbp_c[trainset_idx]
            test_file['label_nsbp_c'] = label_nsbp_c[testset_idx]
            train_file['label_ndbp_c'] = label_ndbp_c[trainset_idx]
            test_file['label_ndbp_c'] = label_ndbp_c[testset_idx]
            train_file['segment'] = data['idx_segment'][trainset_idx]
            test_file['segment'] = data['idx_segment'][testset_idx]
            train_file['segment_average'] = segment_average[trainset_idx]      # 按段全平均校准血压
            test_file['segment_average'] = segment_average[testset_idx]
            train_file['subject_sbp'] = subject_sbp[trainset_idx]          # 校准后所有数据条目的血压值
            test_file['subject_sbp'] = subject_sbp[testset_idx]
            train_file['trainset_idx'] = trainset_idx
            test_file['testset_idx'] = testset_idx


if __name__ == '__main__':
    data = load_mat(mat_file)
    # get_train_and_test_idx(mat_file)
    segment_average = get_segment_average(mat_file)
    split_set_v6(data, segment_average, 'trainset_idx.txt', 'testset_idx.txt')

    with h5py.File('train_set_v8_random_5.h5') as train_f:
        print(len(list(train_f['label_ndbp_c'])))
        with h5py.File('test_set_v8_random_5.h5') as test_f:
            print(list(test_f['testset_idx']))
    '''
    save2h5(a, 'alldata.h5')
    with h5py.File('alldata.h5') as f:
        print(list(f.keys()))
        print(f['ecg'])
    split_dataset(a, 0.8)
    
    with h5py.File('train_set.h5') as train_f:
        print(len(list(train_f['label_ndbp_c'])))
        with h5py.File('test_set.h5') as test_f:
            print(len(list(test_f['label_ndbp_c'])))
    '''
