import tensorflow as tf
import tensorflow.contrib.slim as slim


class Network:
    def __init__(self, session, action_count, resolution, lr, batch_size, trace_length, hidden_size, scope):
        self.session = session
        self.resolution = resolution
        self.train_batch_size = batch_size
        self.trace_length_size = trace_length
        self.action_fc_size = 128
        self.action_count = action_count

        # shape = [8*32,80,45,3]
        self.state = tf.placeholder(tf.float32, shape=[None, resolution[0], resolution[1], resolution[2]])

        # shape = [8*32,3]
        self.act = tf.placeholder(tf.float32, shape=[None, action_count])

        # shape = [8*32,128]
        action_fc = slim.fully_connected(self.act, self.action_fc_size, activation_fn=None)

        # shape = [8*32,19,10,32]
        conv1 = slim.conv2d(inputs=self.state, num_outputs=32, kernel_size=[8, 8], stride=[4, 4],
                            activation_fn=tf.nn.relu, padding='VALID', scope=scope+'_c1')

        # shape = [8*32,8,4,64]
        conv2 = slim.conv2d(inputs=conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2],
                            activation_fn=tf.nn.relu, padding='VALID', scope=scope+'_c2')

        # shape = [8*32,6,2,64]
        conv3 = slim.conv2d(inputs=conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1],
                            activation_fn=tf.nn.relu, padding='VALID', scope=scope+'_c3')

        # shape = [8*32,6*2*64]
        conv3 = tf.reshape(conv3, [-1,6*2*64])

        # shape = [8*32,768+128]
        flat = tf.concat([action_fc,conv3], 1)

        # shape = [8*32*896]
        flat = slim.flatten(flat)

        # hidden_size == 768 self.action_fc_size == 128
        self.cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size+self.action_fc_size, state_is_tuple=True)
        self.train_length = tf.placeholder(dtype=tf.int32)
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])

        # shape = [32,8,896]
        self.fc_reshape = tf.reshape(flat, [self.batch_size, self.train_length, hidden_size+self.action_fc_size])
        self.state_in = self.cell.zero_state(self.batch_size + self.action_fc_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=self.fc_reshape, cell=self.cell, dtype=tf.float32,
                                                     initial_state=self.state_in, scope=scope+'_rnn')
        self.rnn = tf.reshape(self.rnn, shape=[-1, hidden_size+self.action_fc_size])

        self.q = slim.fully_connected(self.rnn, action_count, activation_fn=None)

        self.best_a = tf.argmax(self.q, 1)

        self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, action_count, dtype=tf.float32)
        self.q_chosen = tf.reduce_sum(tf.multiply(self.q, self.actions_onehot), axis=1)

        self.loss = tf.losses.mean_squared_error(self.q_chosen, self.target_q)

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, momentum=0.95, epsilon=0.01)

        self.train_step = self.optimizer.minimize(self.loss)

    def learn(self, last_action, state, target_q, state_in, action):
        act_onehot = []
        for i in range(32*8):
            tmp = [0]*self.action_count
            tmp[last_action[i]] = 1
            act_onehot.append(tmp)
        feed_dict = {self.state: state, self.target_q: target_q, self.train_length: self.trace_length_size,
                     self.batch_size: self.train_batch_size, self.state_in: state_in, self.actions: action, self.act: act_onehot}
        self.session.run(self.train_step, feed_dict=feed_dict)

    def get_q(self, last_action, state, state_in):
        act_onehot = []
        for i in range(32*8):
            tmp = [0]*self.action_count
            tmp[last_action[i]] = 1
            act_onehot.append(tmp)
        return self.session.run(self.q, feed_dict={self.state: state, self.train_length: self.trace_length_size,
                                                   self.batch_size: self.train_batch_size, self.state_in: state_in, self.act: act_onehot})

    def get_best_action(self, last_action, state, state_in):
        act_onehot = [0]*self.action_count
        act_onehot[last_action] = 1
        return self.session.run([self.best_a, self.rnn_state], feed_dict={self.state: [state], self.train_length: 1,
                                                                          self.batch_size: 1, self.state_in: state_in, self.act:[act_onehot]})

    def get_cell_state(self, last_action, state, state_in):
        act_onehot = [0]*self.action_count
        act_onehot[last_action] = 1
        return self.session.run(self.rnn_state, feed_dict={self.state: [state], self.train_length: 1,
                                                           self.state_in: state_in, self.batch_size: 1, self.act:[act_onehot]})
