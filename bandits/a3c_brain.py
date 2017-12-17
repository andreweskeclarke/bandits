import threading
import tensorflow as tf

from keras.models import *
from keras.layers import *
from keras import backend as K


def one_layer_mlp_model(n_inputs, n_actions, n_timesteps):
    l_input = Input(batch_shape=(None, 1, n_inputs))
    l_dense = Dense(8*n_actions, activation='relu')(l_input)

    out_actions = Dense(n_actions, activation='softmax')(l_dense)
    out_value   = Dense(1, activation='linear')(l_dense)

    model = Model(inputs=[l_input], outputs=[out_actions, out_value])
    model._make_predict_function()    # have to initialize before threading
    return model


def two_layer_mlp_model(n_inputs, n_actions, n_timesteps):
    l_input = Input(batch_shape=(None, 1, n_inputs))
    l_dense1 = Dense(8*n_inputs, activation='relu')(l_input)
    l_dense = Dense(8*n_actions, activation='relu')(l_dense1)

    out_actions = Dense(n_actions, activation='softmax')(l_dense)
    out_value   = Dense(1, activation='linear')(l_dense)

    model = Model(inputs=[l_input], outputs=[out_actions, out_value])
    model._make_predict_function()    # have to initialize before threading
    return model


def lstm_model(n_inputs, n_actions, n_timesteps):
    l_input = Input(batch_shape=(None, n_timesteps, n_inputs))
    l_lstm_input = Input(batch_shape=(None, 8*n_inputs))
    l_lstm = GRU(8*n_inputs, activation='relu', return_sequences=True)(l_input, initial_state=l_lstm_input)
    l_dense = Dense(8*n_actions, activation='relu')(l_lstm)

    out_actions = Dense(n_actions, activation='softmax')(l_dense)
    out_value   = Dense(1, activation='linear')(l_dense)

    # Return the full LSTM output sequence - so people can reuse the hidden LSTM state as required
    model = Model(inputs=[l_input, l_lstm_input], outputs=[out_actions, out_value, l_lstm])
    model._make_predict_function()    # have to initialize before threading
    return model


MODELS = {
    'ONE_LAYER_MLP_MODEL': one_layer_mlp_model,
    'TWO_LAYER_MLP_MODEL': two_layer_mlp_model,
    'LSTM_MODEL': lstm_model,
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
            coef_value_loss=0.5,
            coef_entropy_loss=0.01,
            gamma=0.9,
            model_name='TWO_LAYER_MLP_MODEL'):
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

        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model = MODELS[model_name](n_inputs=self.n_inputs, n_actions=self.n_actions, n_timesteps=self.n_timesteps)
        self.graph = self._build_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()
        self.default_graph.finalize()   # avoid modifications

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
        return np.zeros((batch_size, 8*self.n_inputs), dtype=np.float32)

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
            self.training_data = {
                    'observation': list(),
                    'action': list(),
                    'reward': list(),
                    'next_observation': list(),
                    'discount': list(),
                    'mask': list(),
                    }

        s = np.stack(s)
        a = np.stack(a)
        r = np.stack(r)
        s_ = np.stack(s_)
        discount = np.stack(discount)
        s_mask = np.stack(s_mask)
        lstm_s = self.default_state_h(batch_size=len(s))

        if len(s) > 5*self.batch_size: print("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)
        r = r + discount * v * s_mask    # set v to 0 where s_ is terminal state
        
        s_t, a_t, r_t, lstm_s_t, minimize = self.graph
        feed_dict = {s_t: s, a_t: a, r_t: r, lstm_s_t: lstm_s, self.model.inputs[1]: lstm_s}
        self.session.run(minimize, feed_dict=feed_dict)
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
        lstm_s_t = tf.placeholder(tf.float32, shape=(None, 8*self.n_inputs))
        
        p_actions, v, _ = model([s_t, lstm_s_t])

        log_prob = tf.log( tf.reduce_sum(p_actions * a_t, axis=2, keep_dims=True) + 1e-10)
        advantage = r_t - v

        loss_policy = - log_prob * tf.stop_gradient(advantage)                                                             # maximize policy
        loss_value  = self.coef_value_loss * tf.square(advantage)                                                          # minimize value error
        entropy = self.coef_entropy_loss * tf.reduce_sum(p_actions * tf.log(p_actions + 1e-10), axis=2, keep_dims=True)    # maximize entropy (regularization)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=.99)
        minimize = optimizer.minimize(loss_total)

        return s_t, a_t, r_t, lstm_s_t, minimize


