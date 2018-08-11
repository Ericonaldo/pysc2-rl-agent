import os, argparse, pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

from common import Config
from rl.agent import clip_log
from rl.model import fully_conv


# TODO extract this to an agent/ module
class ILAgent:
    def __init__(self, sess, model_fn, config, lr):
        self.sess, self.config, self.lr = sess, config, lr
        (self.policy, self.value), self.inputs = model_fn(config)

        loss_fn, self.loss_inputs = self._loss_func()

        self.step = tf.Variable(0, trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-5)
        # opt = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.99, epsilon=1e-5)
        self.train_op = layers.optimize_loss(loss=loss_fn, optimizer=opt, learning_rate=None, global_step=self.step)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('logs/' + self.config.full_id(), graph=None)

    def train(self, states, actions):
        feed_dict = dict(zip(self.inputs + self.loss_inputs, states + actions))
        result, result_summary, step = self.sess.run([self.train_op, self.summary_op, self.step], feed_dict)

        self.summary_writer.add_summary(result_summary, step)

        return result

    def _loss_func(self):
        actions = [tf.placeholder(tf.int32, [None]) for _ in self.policy]
        acts = [tf.one_hot(actions[i], d) for i, (d, _) in enumerate(self.config.policy_dims())]
        ce = sum([-tf.reduce_sum(a * clip_log(p), axis=-1) for a, p in zip(acts, self.policy)])
        ce_loss = tf.reduce_mean(ce)
        val_loss = 0 * tf.reduce_mean(self.value) # hack to match a2c agent computational graph
        return ce_loss + val_loss, actions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sz", type=int, default=32)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--batch_sz', type=int, default=128)
    parser.add_argument("--map", type=str, default='MoveToBeacon')
    parser.add_argument("--cfg_path", type=str, default='config.json.dist')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    tf.reset_default_graph()
    sess = tf.Session()

    config = Config(args.sz, args.map, "il")
    os.makedirs('weights/' + config.full_id(), exist_ok=True)
    cfg_path = 'weights/%s/config.json' % config.full_id()
    config.build(args.cfg_path)
    config.save(cfg_path)

    with open('replays/%s.pkl' % config.map_id(), 'rb') as fl:
        rollouts = pickle.load(fl)
    for i in range(2):
        for j in range(len(rollouts[i])):
            rollouts[i][j] = np.array(rollouts[i][j])

    agent = ILAgent(sess, fully_conv, config, args.lr)

    n = len(rollouts[0][0])
    epochs = max(1, args.samples // n)
    n_batches = min(args.samples, n) // args.batch_sz + 1
    print("n_samples %d, epochs: %d, batches: %d" % (n, epochs, n_batches))
    for _ in range(epochs):
        for _ in range(n_batches):
            idx = np.random.choice(n, args.batch_sz, replace=False)
            sample = [s[idx] for s in rollouts[0]], [a[idx] for a in rollouts[1]]
            res = agent.train(*sample)
            print(res)
    agent.saver.save(sess, 'weights/%s/a2c' % config.full_id(), global_step=agent.step)
