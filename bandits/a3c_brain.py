import os
import datetime as dt
import threading
import tensorflow as tf
import random

from keras.models import *
from keras.layers import *
from keras import backend as K


def lstm_model(n_inputs, n_actions, n_timesteps):
    l_input = Input(batch_shape=(None, n_timesteps, n_inputs))
    l_lstm_input_1 = Input(batch_shape=(None, 48))
    l_lstm_input_2 = Input(batch_shape=(None, 48))
    l_lstm = LSTM(48, return_sequences=True, activation='tanh', recurrent_activation='tanh')(l_input, initial_state=(l_lstm_input_1, l_lstm_input_2))

    out_actions = Dense(n_actions, activation='softmax')(l_lstm)
    out_value   = Dense(1, activation='linear')(l_lstm)

    model = Model(inputs=[l_input, l_lstm_input_1, l_lstm_input_2], outputs=[out_actions, out_value, l_lstm])
    model._make_predict_function()    # have to initialize before threading
    return model


def gru_model(n_inputs, n_actions, n_timesteps):
    l_input = Input(batch_shape=(None, n_timesteps, n_inputs))
    l_gru_input = Input(batch_shape=(None, 48))
    l_gru = CuDNNGRU(48, return_sequences=True)(l_input, initial_state=l_gru_input)

    tf.summary.histogram('l_gru', l_gru)

    out_actions = Dense(n_actions, activation='softmax')(l_gru)
    out_value   = Dense(1, activation='linear')(l_gru)

    model = Model(inputs=[l_input, l_gru_input], outputs=[out_actions, out_value, l_gru])
    model._make_predict_function()    # have to initialize before threading
    return model


MODELS = {
    'LSTM_MODEL': lstm_model,
    'GRU_MODEL': gru_model,
    }


# Inspired from:
# https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py
# and the corresponding article at:
# https://jaromiru.com/2017/03/26/lets-make-an-a3c-implementation/

class A3CBrain(object):
    def __init__(self,
            n_actions,
            n_inputs,
            n_timesteps,
            learning_rate=5e-3,
            batch_size=32,
            coef_value_loss=0.05,
            coef_entropy_loss=0.01,
            gamma=0.9,
            model_name='TWO_LAYER_MLP_MODEL',
            n_look_ahead=3):
        self.n_actions = n_actions
        self.n_inputs = n_inputs
        self.n_timesteps = n_timesteps
        self.multi_threading_lock = threading.Lock()
        self.training_data = {
                'observation': list(),
                'action': list(),
                'reward': list(),
                'next_observation': list(),
                'discount': list(),
                'mask': list(),
                }
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.coef_value_loss = coef_value_loss
        self.coef_entropy_loss = coef_entropy_loss
        self.gamma = gamma
        self._n_optimize_runs = 0
        self.avg_reward = 0.0
        self.n_look_ahead = n_look_ahead

        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model = MODELS[model_name](n_inputs=self.n_inputs, n_actions=self.n_actions, n_timesteps=self.n_timesteps)
        self.graph = self._build_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()
        self.default_graph.finalize()   # avoid modifications

    def update_coef_entropy_loss(self, coef):
        self.coef_entropy_loss = max(0.0, coef)

    def reset(self):
        self.training_data = {
                'observation': list(),
                'action': list(),
                'reward': list(),
                'next_observation': list(),
                'discount': list(),
                'mask': list(),
                }

    def single_prediction(self, s, state_h):
        s_input = np.zeros((1, self.n_timesteps, self.n_inputs))
        if s is not None:
            s_input[0,0,:] = s.reshape((self.n_inputs,))
        if state_h is None:
            state_h = self.default_state_h(batch_size=1)
        p, v, state_h = self.model.predict([s_input, state_h])
        p = p[0,0,:].reshape((self.n_actions,))
        v = v[0,0,:].reshape((1,))
        state_h = state_h[0,0,:].reshape((1,-1))
        return p, v, state_h

    def predict(self, s, state_h=None):
        with self.default_graph.as_default():
            if state_h is None:
                batch_size = s.shape[0]
                state_h = self.default_state_h(batch_size=batch_size)
            p, v, state_h = self.model.predict([s, state_h])
            return p, v, state_h

    def default_state_h(self, batch_size=1):
        # TODO how to dunamically figure this out
        return np.zeros((batch_size, 48), dtype=np.float32)

    def predict_v(self, s, state_h=None):
        return self.predict(s, state_h)[1]

    def optimize(self):
        if len(self.training_data['observation']) < self.batch_size:
            return

        with self.multi_threading_lock:
            if len(self.training_data['observation']) < self.batch_size:
                return

            s = self.training_data['observation']
            a = self.training_data['action']
            r = self.training_data['reward']
            s_ = self.training_data['next_observation']
            discount = self.training_data['discount']
            s_mask = self.training_data['mask']

            # Clear training data
            self.reset()

        s = np.stack(s)
        a = np.stack(a)
        r = np.stack(r)
        s_ = np.stack(s_)
        discount = np.stack(discount)
        s_mask = np.stack(s_mask)
        lstm_s = self.default_state_h(batch_size=len(s))

        if len(s) > 5*self.batch_size: print("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)
        v_pred = np.multiply(np.multiply(discount, v), s_mask) # set v to 0 where s_ is terminal state
        r_target = r + v_pred

        s_t, a_t, r_t, r_obs, lstm_s_t, avg_reward, minimize, summaries = self.graph
        feed_dict = {
                s_t: s,
                a_t: a,
                r_t: r_target,
                r_obs: r,
                lstm_s_t: lstm_s,
                avg_reward: np.array([self.avg_reward]),
                self.model.inputs[1]: lstm_s,
                self.model.inputs[0]: s}
        self.session.run(minimize, feed_dict=feed_dict)
        s = self.session.run(summaries, feed_dict=feed_dict)
        self.train_writer.add_summary(s, self._n_optimize_runs)
        self._n_optimize_runs += 1

    def push_training_episode(self, observation, action, reward, next_observation, discount, mask):
        with self.multi_threading_lock:
            self.training_data['observation'].append(observation)
            self.training_data['action'].append(action)
            self.training_data['reward'].append(reward)
            self.training_data['next_observation'].append(next_observation)
            self.training_data['discount'].append(discount)
            self.training_data['mask'].append(mask)

    def _build_graph(self, model):
        s_t = tf.placeholder(tf.float32, shape=(None, self.n_timesteps, self.n_inputs))
        a_t = tf.placeholder(tf.float32, shape=(None, self.n_timesteps, self.n_actions))
        r_t = tf.placeholder(tf.float32, shape=(None, self.n_timesteps, 1)) # discounted n step reward
        r_obs = tf.placeholder(tf.float32, shape=(None, self.n_timesteps, 1)) # observed rewards, for debugging
        lstm_s_t = tf.placeholder(tf.float32, shape=(None, 48))
        avg_reward = tf.placeholder(tf.float32, shape=(1,))

        p_actions, v, _ = model([s_t, lstm_s_t])

        log_prob = tf.log( tf.reduce_sum( tf.multiply(p_actions, a_t), axis=2, keep_dims=True) + 1e-10)
        advantage = r_t - v

        loss_policy = tf.multiply(-log_prob, tf.stop_gradient(advantage))
        loss_value  = self.coef_value_loss * tf.square(advantage)
        loss_entropy = self.coef_entropy_loss * tf.reduce_sum(p_actions * tf.log(p_actions + 1e-10), axis=2, keep_dims=True)

        loss_total = tf.reduce_mean(loss_policy + loss_value + loss_entropy)

        # optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=.99)
        # minimize = optimizer.minimize(loss_total)
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # minimize = optimizer.minimize(loss_total)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(loss_total))
        for g in gradients:
            tf.summary.histogram(g.name, g)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        for g in clipped_gradients:
            tf.summary.histogram('clipped_' + g.name, g)
        minimize = optimizer.apply_gradients(zip(clipped_gradients, variables))

        tf.summary.scalar('loss_policy', tf.reduce_mean(loss_policy))
        tf.summary.scalar('loss_value', tf.reduce_mean(loss_value))
        tf.summary.scalar('loss_entropy', tf.reduce_mean(loss_entropy))
        tf.summary.scalar('loss', loss_total)
        tf.summary.scalar('reward target', tf.reduce_mean(r_t))
        tf.summary.scalar('reward observed', tf.reduce_mean(r_obs))
        tf.summary.scalar('avg test score', tf.reduce_mean(avg_reward))
        tf.summary.scalar('value', tf.reduce_mean(v))
        tf.summary.histogram('value histogram', tf.reduce_mean(v))
        summaries = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.tensor_board_directory(), self.session.graph)

        return s_t, a_t, r_t, r_obs, lstm_s_t, avg_reward, minimize, summaries

    def tensor_board_directory(self):
        datetime_dir = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        # tf_path = os.path.join('/home/andrew/src/bandits/training_reports/', datetime_dir)
        tf_path = os.path.join('/tmp/training_reports/', datetime_dir)
        os.makedirs(tf_path)
        return tf_path


