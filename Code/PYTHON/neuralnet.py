import numpy as np
import utils
import pandas as pd
import tensorflow as tf

def neuralnet(data_dict, **kwargs):
    R,T,A,L = data_dict['Xtrain'].shape
    Rtest,Ttest = data_dict['Ytest'].shape
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(A,L)),
        tf.keras.layers.Dense(A*L+T, activation=tf.nn.tanh),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(A*L+T, activation=tf.nn.tanh),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(A, activation=tf.nn.softmax)
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(data_dict['Xtrain'].reshape(R*T,A,L), data_dict['Ytrain'].flatten()-1, epochs=30, batch_size=T)
    test_loss, test_acc = model.evaluate(data_dict['Xtest'].reshape(Rtest*Ttest,A,L), data_dict['Ytest'].flatten()-1)
    return test_acc


def nn_demo(dataset):
    path_to_data = "../../Data/{0}/".format(dataset)
    data = utils.get_data(path_to_data, holdout=1)

    result = neuralnet(data)
    return result

if __name__ == "__main__":
    print("ACCURACY:", nn_demo("R1_PremiumChocolate"))
