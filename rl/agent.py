import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

# 定义并实现了agent的行为，运行在runner里，一个agent与多个环境进行交互

class A2CAgent:
    def __init__(self, sess, model_fn, config, restore=False, discount=0.99, lr=1e-4, vf_coef=0.25, ent_coef=1e-3, clip_grads=1.):
        self.sess, self.config, self.discount = sess, config, discount
        self.vf_coef, self.ent_coef = vf_coef, ent_coef

        (self.policy, self.value), self.inputs = model_fn(config) # policy[0]是one_hot动作函数，shape = [None, 524];  policy[1:]为13个one_hot参数shape = [[None, dim1], [None, dim2],...], spatial的feature维度是1024
        self.action = [sample(p) for p in self.policy] # action 返回的是数值
            
        with tf.variable_scope('loss'):
            loss_fn, self.loss_inputs = self._loss_func()
        
        with tf.variable_scope('train'):
            self.step = tf.Variable(0, trainable=False)
            opt = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.99, epsilon=1e-5)
            # opt = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-5)
            self.train_op = layers.optimize_loss(loss=loss_fn, optimizer=opt, learning_rate=None, global_step=self.step, clip_gradients=clip_grads)
            self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        if restore:
            self.saver.restore(self.sess, tf.train.latest_checkpoint('weights/' + self.config.full_id()))

        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('logs/' + self.config.full_id(), graph=None)
        self.summary_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START), sess.run(self.step))

    # TODO: get rid of the step param; gracefully restore for console logs as well
    def train(self, step, states, actions, rewards, dones, last_value, ep_rews):
        if step % 500 == 0:
            self.saver.save(self.sess, 'weights/%s/a2c' % self.config.full_id(), global_step=self.step)

        returns = self._compute_returns(rewards, dones, last_value)

        feed_dict = dict(zip(self.inputs + self.loss_inputs, states + actions + [returns])) # 你非得写成这样令人看了得转个弯才能看懂吗
        result, result_summary, step = self.sess.run([self.train_op, self.summary_op, self.step], feed_dict) # 每次训练都记录summary

        self.summary_writer.add_summary(result_summary, step)
        self.summary_writer.add_summary(summarize(rewards=ep_rews), step) # 每次训练添加数据到tensorboard， ep_rews是每个env在一个episode内（n_step）平均的reward

        return result

    def act(self, state):
        return self.sess.run([self.action, self.value], feed_dict=dict(zip(self.inputs, state)))

    def get_value(self, state):
        return self.sess.run(self.value, feed_dict=dict(zip(self.inputs, state)))

    def _loss_func(self):
        returns = tf.placeholder(tf.float32, [None])
        actions = [tf.placeholder(tf.int32, [None]) for _ in range(len(self.policy))] # policy is a list, actions is a list  每个元素对应动作函数或参数

        adv = tf.stop_gradient(returns - self.value)
        logli = sum([clip_log(select(a, p)) for a, p in zip(actions, self.policy)])
        entropy = sum([-tf.reduce_sum(p * clip_log(p), axis=-1) for p in self.policy])

        policy_loss = -tf.reduce_mean(logli * adv)
        entropy_loss = -self.ent_coef * tf.reduce_mean(entropy)
        value_loss = self.vf_coef * tf.reduce_mean(tf.square(returns - self.value))

        tf.summary.scalar('loss/policy', policy_loss)
        tf.summary.scalar('loss/entropy', entropy_loss)
        tf.summary.scalar('loss/value', value_loss)

        return policy_loss + entropy_loss + value_loss, actions + [returns]

    def _compute_returns(self, rewards, dones, last_value):
        returns = np.zeros((dones.shape[0]+1, dones.shape[1]))
        returns[-1] = last_value
        for t in reversed(range(dones.shape[0])):
            returns[t] = rewards[t] + self.discount * returns[t+1] * (1-dones[t])
        returns = returns[:-1]
        return returns.flatten()


def select(acts, policy): # 传入的policy.shape = [batch, dim_i], acts.shape=[batch]
    return tf.gather_nd(policy, tf.stack([tf.range(tf.shape(policy)[0]), acts], axis=1)) # 把act对应的policy的值取出来


# based on https://github.com/pekaalto/sc2aibot/blob/master/common/util.py#L5-L11
def sample(probs):
    u = tf.random_uniform(tf.shape(probs))
    return tf.argmax(tf.log(u) / probs, axis=1)


def clip_log(probs):
    return tf.log(tf.clip_by_value(probs, 1e-12, 1.0))


def summarize(**kwargs):
    summary = tf.Summary()
    for k, v in kwargs.items():
        summary.value.add(tag=k, simple_value=v)
    return summary
