import tensorflow as tf
import numpy as np
from collections import Counter
from string import punctuation

#data preprocess
with open("./input/reviews.txt", 'r') as f:
    reviews = f.read()
with open("./input/labels.txt", 'r') as f:
    labels = f.read()

all_text = ''.join([c for c in reviews if c not in punctuation])
reviews = all_text.split(('\n'))

all_text = ' '.join(reviews)
words = all_text.split()

counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

reviews_int = []
for each in reviews:
    reviews_int.append([vocab_to_int[i] for i in each.split()])

labels = labels.split('\n')
labels = np.array([1 if each == 'positive' else 0 for each in labels])
num_classes = len(np.unique(labels))
print ("num_classes = ", num_classes)

non_zero_index = [ii for ii, review in enumerate(reviews_int) if len(review) > 0]
reviews_int = [reviews_int[ii] for ii in non_zero_index]
labels = np.array([labels[ii] for ii in non_zero_index])

seq_len = 200
features = np.zeros((len(reviews_int), seq_len), dtype=int)
for i, row in enumerate(reviews_int):
    features[i, -len(row):] = np.array(row)[:seq_len]

split_frac = 0.8
split_idx = int(len(features)*0.8)
train_x, test_x = features[:split_idx], features[split_idx:]
train_y, test_y = labels[:split_idx], labels[split_idx:]

# model
lstm_size = 64
lstm_layers = 3
batch_size = 100
learning_rate = 0.001

n_words = len(vocab_to_int)+1

def get_batches(x, y, batch_size=100):
    n_batches = len(x) // batch_size
    x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii + batch_size], y[ii:ii + batch_size]

graph = tf.Graph()
embed_size = 100
with graph.as_default():
    inputs_ = tf.placeholder(tf.int32, [None, None], name="inputs")
    labels_ = tf.placeholder(tf.int32, [None, None], name="labels")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs_)
    def get_a_cell(lstm_size, keep_prob):
        lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
        drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop
    cell = tf.nn.rnn_cell.MultiRNNCell(
        [get_a_cell(lstm_size, keep_prob) for _ in range(lstm_layers)]
    )
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)
    preds = tf.contrib.layers.fully_connected(outputs[:, -1], num_classes, activation_fn=tf.sigmoid)
    labels_v = tf.one_hot(labels_, depth=num_classes)
    labels_one_hot = tf.reshape(labels_v, [-1, num_classes])
    cost = tf.losses.log_loss(labels_one_hot, preds)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    correct_pred = tf.equal(tf.argmax(preds, 1), tf.argmax(labels_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

epochs = 30
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    for e in range(epochs):
        state = sess.run(initial_state)
        for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
            feed = {inputs_: x,
                    labels_: y[:, None],
                    keep_prob: 0.5,
                    initial_state: state}
            loss, state, _, accu = sess.run([cost, final_state, optimizer, accuracy], feed_dict=feed)
            if iteration % 5 == 0:
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Train loss: {:.3f}".format(loss),
                      "Accuracy: {:.3f}".format(accu))

            if iteration % 25 == 0:
                val_acc = []
                val_loss = []
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                for x, y in get_batches(test_x, test_y, batch_size):
                    feed = {inputs_: x,
                            labels_: y[:, None],
                            keep_prob: 1,
                            initial_state: val_state}
                    batch_loss, batch_acc, val_state = sess.run([cost, accuracy, final_state], feed_dict=feed)
                    val_acc.append(batch_acc)
                    val_loss.append(batch_loss)
                print("Val acc: {:.3f}".format(np.mean(val_acc)), "Train loss: {:.3f}".format(batch_loss))
            iteration += 1
exit(0)