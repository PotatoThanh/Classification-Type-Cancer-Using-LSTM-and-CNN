import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences

NUM_TRAIN = 330000
NUM_VALID = 1000

def preprocess_data(file_path):
    def normalize_data(data):
        data_min = np.min(data)
        data_max = np.max(data)

        norm_data = (data-data_min)/(data_max - data_min)

        return norm_data

    def paddingSequence(feature, maxlen):
        str_features = []
        n = feature.shape[0]

        for i in range(0, n):
            x = feature[i]
            temp = x.split(",")
            str_features.append(np.array(temp).astype(int))

        int_features = np.array(str_features)
        return pad_sequences(int_features, padding='post', maxlen=maxlen)

    def createFeatures(data):
        # generate features

        Data = data[:, 0:data.shape[1]-2]
        Data = np.array(Data).astype(float)
        for i in range(1, data.shape[1] - 2):
            Data[:, i] = normalize_data(Data[:, i])

        # print Data
        # features_dict = data[:, 1:6]
        Reference_Allele = data[:, 7]
        Reference_Allele = paddingSequence(Reference_Allele, maxlen=96)

        Tumor_Allele = data[:, 8]
        Tumor_Allele = paddingSequence(Tumor_Allele, maxlen=82)

        y = keras.utils.to_categorical(np.array(Data[:, 0]).astype(int))
        x = np.reshape(Data[:, 1:data.shape[1]-2], (-1, 6, 1))
        return y, x, Reference_Allele, Tumor_Allele

    data = np.load(file_path)

    #SHUFFLE DATA
    data = np.take(data, np.random.permutation(data.shape[0]), axis=0, out=data)

    # print data

    NUM_TEST = data.shape[0] - NUM_TRAIN - NUM_VALID

    train_data = data[0:NUM_TRAIN, :]
    valid_data = data[NUM_TRAIN:NUM_TRAIN+NUM_VALID, :]
    test_data = data[NUM_TRAIN+NUM_VALID:, :]

    #
    [y_train, x_train_feature, x_train_ref, x_train_tumor] = createFeatures(train_data)
    [y_valid, x_valid_feature, x_valid_ref, x_valid_tumor] = createFeatures(valid_data)
    [y_test, x_test_feature, x_test_ref, x_test_tumor] = createFeatures(test_data)

    return (y_train, x_train_feature, x_train_ref, x_train_tumor,
            y_valid, x_valid_feature, x_valid_ref, x_valid_tumor,
            y_test, x_test_feature, x_test_ref, x_test_tumor)




