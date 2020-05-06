import pickle
import tensorflow as tf
import tensorflow.keras as kr
import numpy as np
import random
import pandas as pd
from sklearn.metrics import confusion_matrix


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
hidden_dim = 128
dropout_keep_prob = 0.5
learning_rate = 1e-3
batch_size = 32
steps = 5000
print_per_batch = steps / 20


def build_model():
    tf.reset_default_graph()
    # shape=(None, 600)
    X_holder = tf.placeholder(tf.int32, [None, seq_length])
    # shape=(None, 2)
    Y_holder = tf.placeholder(tf.float32, [None, class_number])
    embedding = tf.get_variable('embedding', [size, embedding_dim])
    embedding_inputs = tf.nn.embedding_lookup(embedding, X_holder)
    con = tf.layers.conv1d(embedding_inputs, num_filters, kernel_size)
    max_pooling = tf.reduce_max(con, reduction_indices=[1])
    full_connect = tf.layers.dense(max_pooling, hidden_dim)
    full_connect_dropout = tf.contrib.layers.dropout(full_connect, keep_prob=dropout_keep_prob)
    full_connect_activate = tf.nn.relu(full_connect_dropout)
    soft_max = tf.layers.dense(full_connect_activate, class_number)
    predict_Y = tf.nn.softmax(soft_max)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_holder, logits=soft_max)
    loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    predict_y = tf.argmax(predict_Y, 1)
    isCorrect = tf.equal(tf.argmax(Y_holder, 1), predict_y)
    accuracy = tf.reduce_mean(tf.cast(isCorrect, tf.float32))

    return X_holder, Y_holder, train, loss, accuracy, predict_y


def content_id_list(content):
    return [word_id_list[word] for word in content if word in word_id_list]


def content_2x(content_list):
    id_list_list = [content_id_list(content[0]) for content in content_list]
    X = kr.preprocessing.sequence.pad_sequences(id_list_list, seq_length)
    return X


def label_2y(label_list):
    y = label_encoder.transform(label_list)
    Y = kr.utils.to_categorical(y, class_number)
    return Y


def train_model():

    X_holder, Y_holder, train, loss, accuracy, predict_y = build_model()

    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)
    train_X = content_2x(x_train)
    train_Y = label_2y(y_train[:, 0])
    test_X = content_2x(x_test)
    test_Y = label_2y(y_test[:, 0])

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
            loss_value, accuracy_value = session.run([loss, accuracy], {X_holder: batch_X, Y_holder: batch_Y})

            print('step:%d loss:%.4f accuracy:%.4f' % (step, loss_value, accuracy_value))

    show_confusion_matrix(session, predict_y, X_holder)


def predict(content_list, session, predict_y, x_holder):
    if type(content_list) == str:
        content_list = [content_list]
    batch_X = content_2x(content_list)
    predict_y = session.run(predict_y, {x_holder: batch_X})
    predict_label_list = label_encoder.inverse_transform(predict_y)
    return predict_label_list


def predict_all(session, predict_y, x_holder):
    predict_label_list = []
    _batch_size = 100
    for i in range(0, len(x_test), _batch_size):
        content_list = x_test[i: i + _batch_size]
        predict_label = predict(content_list, session, predict_y, x_holder)
        predict_label_list.extend(predict_label)
    return predict_label_list


def show_confusion_matrix(session, predict_y, x_holder):
    predict_label_list = predict_all(session, predict_y, x_holder)
    df = pd.DataFrame(confusion_matrix(y_test[:, 0].tolist(), predict_label_list),
                      columns=label_encoder.classes_,
                      index=label_encoder.classes_)
    print('\n Confusion Matrix:')
    print(df)


def main():
    train_model()


if __name__ == '__main__':
    main()
