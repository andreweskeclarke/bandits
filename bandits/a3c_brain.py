import threading
import tensorflow as tf

from keras.models import *
from keras.layers import *
from keras import backend as K


def one_layer_mlp_model(n_inputs, n_actions):
    l_input = Input(batch_shape=(None, n_inputs))
    l_dense = Dense(8*n_actions, activation='relu')(l_input)

    out_actions = Dense(n_actions, activation='softmax')(l_dense)
    out_value   = Dense(1, activation='linear')(l_dense)

    model = Model(inputs=[l_input], outputs=[out_actions, out_value])
    model._make_predict_function()    # have to initialize before threading
    return model


def two_layer_mlp_model(n_inputs, n_actions):
    l_input = Input(batch_shape=(None, n_inputs))
    l_dense1 = Dense(8*n_inputs, activation='relu')(l_input)
    l_dense = Dense(8*n_actions, activation='relu')(l_dense1)

    out_actions = Dense(n_actions, activation='softmax')(l_dense)
    out_value   = Dense(1, activation='linear')(l_dense)

    model = Model(inputs=[l_input], outputs=[out_actions, out_value])
    model._make_predict_function()    # have to initialize before threading
    return model


MODELS = {
    'ONE_LAYER_MLP_MODEL': one_layer_mlp_model,
    'TWO_LAYER_MLP_MODEL': two_layer_mlp_model,
    }


# Inspired from:
# https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py
# and the corresponding article at:
# https://jaromiru.com/2017/03/26/lets-make-an-a3c-implementation/

class A3CBrain(object):
    def __init__(self,
            n_actions,
            n_inputs,
            learning_rate=5e-3,
            batch_size=32,
            coef_value_loss=0.5,
            coef_entropy_loss=0.01,
            gamma=0.9,
            model_name='TWO_LAYER_MLP_MODEL'):
        self.n_actions = n_actions
        self.n_inputs = n_inputs
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

        self.model = MODELS[model_name](self.n_inputs, self.n_actions)
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

    def predict(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(self._prep_input(s))
            return p, v

    def _prep_input(self, s):
        if s is None:
            s = np.zeros(self.n_actions)
        return s.reshape((-1, self.n_actions))

    def predict_action(self, s):
        return self.predict(s)[0]

    def predict_v(self, s):
        return self.predict(s)[1]

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

        s = np.vstack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        discount = np.vstack(discount)
        s_mask = np.vstack(s_mask)

        if len(s) > 5*self.batch_size: print("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)
        r = r + discount * v * s_mask    # set v to 0 where s_ is terminal state
        
        s_t, a_t, r_t, minimize = self.graph
        self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})
        self._n_optimize_runs += 1


    def push_training_example(self, observation, action, reward, next_observation, discount):
        with self.multi_threading_lock:
            mask = 0.0 if next_observation is None else 1.0
            self.training_data['observation'].append(self._prep_input(observation))
            self.training_data['action'].append(action)
            self.training_data['reward'].append(reward)
            self.training_data['next_observation'].append(self._prep_input(next_observation))
            self.training_data['discount'].append(discount)
            self.training_data['mask'].append(mask)

    def _build_graph(self, model):
        s_t = tf.placeholder(tf.float32, shape=(None, self.n_inputs))
        a_t = tf.placeholder(tf.float32, shape=(None, self.n_actions))
        r_t = tf.placeholder(tf.float32, shape=(None, 1)) # discounted n step reward
        
        p_actions, v = model(s_t)

        log_prob = tf.log( tf.reduce_sum(p_actions * a_t, axis=1, keep_dims=True) + 1e-10)
        advantage = r_t - v

        loss_policy = - log_prob * tf.stop_gradient(advantage)                                                             # maximize policy
        loss_value  = self.coef_value_loss * tf.square(advantage)                                                          # minimize value error
        entropy = self.coef_entropy_loss * tf.reduce_sum(p_actions * tf.log(p_actions + 1e-10), axis=1, keep_dims=True)    # maximize entropy (regularization)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=.99)
        minimize = optimizer.minimize(loss_total)

        return s_t, a_t, r_t, minimize


