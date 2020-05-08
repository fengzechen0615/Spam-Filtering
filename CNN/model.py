import pickle
import tensorflow as tf
import tensorflow.keras as kr
import numpy as np
import random
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# START #
# load parameter
path = "../data/data.pickle"
fr = open(path, "rb")
para = pickle.load(fr)
x_train = para.get("x_train")
x_test = para.get("x_test")
y_train = para.get("y_train")
y_test = para.get("y_test")
data = para.get("data")
label = para.get("label")
word_list = para.get("word_list")
label_encoder = para.get("label_encoder")
word_id_list = para.get("word_id_list")
seq_length = para.get("seq_length")
size = para.get("size")
class_number = np.unique(label).shape[0]
embedding_dim = 64
num_filters = 256
kernel_size = 5
dropout_keep_prob = 0.5
learning_rate = 2e-4
batch_size = 32
steps = 5000
print_per_batch = steps / 20
# END #


def build_model():
    tf.reset_default_graph()
    X_holder = tf.placeholder(tf.int32, [None, seq_length])
    Y_holder = tf.placeholder(tf.float32, [None, class_number])
    embedding = tf.get_variable('embedding', [size, embedding_dim])
    embedding_inputs = tf.nn.embedding_lookup(embedding, X_holder)
    con = tf.layers.conv1d(embedding_inputs, num_filters, kernel_size)
    max_pooling = tf.reduce_max(con, reduction_indices=[1])
    # START #
    # first fully connect layer
    full_connect_1 = tf.layers.dense(max_pooling, 256)
    full_connect_dropout_1 = tf.contrib.layers.dropout(full_connect_1, keep_prob=dropout_keep_prob)
    full_connect_activate_1 = tf.nn.relu(full_connect_dropout_1)
    # second fully connect layer
    full_connect_2 = tf.layers.dense(full_connect_activate_1, 32)
    full_connect_dropout_2 = tf.contrib.layers.dropout(full_connect_2, keep_prob=dropout_keep_prob)
    full_connect_activate_2 = tf.nn.relu(full_connect_dropout_2)
    # third full connect layer
    soft_max = tf.layers.dense(full_connect_activate_2, class_number)
    predict_Y = tf.nn.softmax(soft_max)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_holder, logits=soft_max)
    # END #
    # build loss
    loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    predict_y = tf.argmax(predict_Y, 1)
    correct = tf.equal(tf.argmax(Y_holder, 1), predict_y)
    # build accuracy
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    return X_holder, Y_holder, train, loss, accuracy, predict_y


# START #
def content_id_list(content):
    return [word_id_list[word] for word in content if word in word_id_list]


def content_2x(content_list):
    id_list_list = [content_id_list(content) for content in content_list]
    X = kr.preprocessing.sequence.pad_sequences(id_list_list, seq_length)
    return X


def label_2y(label_list):
    y = label_encoder.transform(label_list)
    Y = kr.utils.to_categorical(y, class_number)
    return Y
# END #


def train_model():

    X_holder, Y_holder, train, loss, accuracy, predict_y = build_model()

    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)

    train_X = content_2x(x_train)
    train_Y = label_2y(y_train)
    test_X = content_2x(x_test)
    test_Y = label_2y(y_test)
    # START #
    accuracy_list = []
    loss_list = []
    step_list = []
    for i in range(steps):
        selected_index = random.sample(list(range(len(train_Y))), k=batch_size)
        batch_X = train_X[selected_index]
        batch_Y = train_Y[selected_index]
        session.run(train, {X_holder: batch_X, Y_holder: batch_Y})
        step = i + 1
        if step % print_per_batch == 0 or step == 1:
            selected_index = random.sample(list(range(len(test_Y))), k=200)
            batch_X = test_X[selected_index]
            batch_Y = test_Y[selected_index]
            _loss, _accuracy = session.run([loss, accuracy], {X_holder: batch_X, Y_holder: batch_Y})
            accuracy_list.append(_accuracy)
            loss_list.append(_loss)
            step_list.append(step)

            print('step:%d loss:%.4f accuracy:%.4f' % (step, _loss, _accuracy))

    # draw plot
    plt.plot(step_list, loss_list, 'r--', label="loss")
    plt.plot(step_list, accuracy_list, 'g--', label="accuracy")
    plt.plot(step_list, loss_list, 'ro-', step_list, accuracy_list, 'g+-')
    plt.xlabel("step")
    plt.ylabel("rate")
    plt.legend()
    plt.show()
    show_confusion_matrix(session, predict_y, X_holder)
    # END #


# START #
def predict_all(session, predict_y, x_holder):
    predict_label_list = []
    _batch_size = 100
    for i in range(0, len(x_test), _batch_size):
        content_list = x_test[i: i + _batch_size]
        if type(content_list) == str:
            content_list = [content_list]
        batch_X = content_2x(content_list)
        _predict_y = session.run(predict_y, {x_holder: batch_X})
        predict_label = label_encoder.inverse_transform(_predict_y)
        predict_label_list.extend(predict_label)
    return predict_label_list


def show_confusion_matrix(session, predict_y, x_holder):
    predict_label_list = predict_all(session, predict_y, x_holder)
    cm = pd.DataFrame(confusion_matrix(y_test.tolist(), predict_label_list),
                      columns=label_encoder.classes_,
                      index=label_encoder.classes_)
    print('\n Confusion Matrix:')
    print(cm)


def main():
    train_model()


if __name__ == '__main__':
    main()
# END #
