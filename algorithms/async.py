import tensorflow as tf
import util.tensor
import util.debug
import algorithms.policy_gradient as pg
import abc


class AsyncAlgorithm:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def needs_update(self, is_terminal):
        pass

    @abc.abstractmethod
    def get_update_gradients(self):
        pass

    @abc.abstractmethod
    def get_update_feed_dict(self):
        pass

    @abc.abstractmethod
    def update_with_gradients_ops(self, gradients):
        pass

    @abc.abstractmethod
    def add_sample(self, state, action, next_state, reward, is_terminal):
        pass


class Async:
    class MainInstance:
        def __init__(self, build_algorithm):
            self._scope_name = 'async_main'
            with tf.variable_scope(self._scope_name, reuse=None):
                self.instance = build_algorithm()
            self._update_from_instance = {}
            self._copy_ops = {}

        def register_instance(self, async_instance):
            self._update_from_instance[async_instance] = \
                self.instance.update_with_gradients_ops(self._redirect_gradients(async_instance))
            #self._update_from_instance[async_instance][0] = util.debug.print_gradient(self._update_from_instance[async_instance][0], self._redirect_gradients(async_instance)[0], message=self._scope_name)
            self._copy_ops[async_instance] = util.tensor.copy_parameters(self._scope_name, async_instance.scope_name)

        def apply_gradient(self, async_instance, feed_dict):
            tf.get_default_session().run(self._update_from_instance[async_instance], feed_dict=feed_dict)

        def copy_parameters(self, async_instance):
            tf.get_default_session().run(self._copy_ops[async_instance])

        def _redirect_gradients(self, async_instance):
            result = []
            for gradients in async_instance.gradients:
                gradients = [(grad, var.name[len(async_instance.scope_name) + 1:].rsplit(':', 1)[0])
                             for grad, var in gradients]
                with tf.variable_scope(self._scope_name, reuse=True):
                    result.append([(grad, tf.get_variable(name)) for grad, name in gradients])
            return result

    class AsyncInstance:
        def __init__(self, build_algorithm, instance_id):
            self.scope_name = 'async_%d' % instance_id
            with tf.variable_scope(self.scope_name, reuse=None):
                self._instance = build_algorithm()
                self.gradients = self._instance.get_update_gradients()
            self._id = instance_id
            self._gradients_to_apply = []
            self._initialized = False

        def update(self, main_instance, state, action, reward, next_state, is_terminal):
            self._instance.add_sample(state, action, reward, next_state, is_terminal)
            if self._instance.needs_update(is_terminal):
                main_instance.apply_gradient(self, self._instance.get_update_feed_dict())
                main_instance.copy_parameters(self)

        def get_action(self, main_instance, state):
            if not self._initialized:
                main_instance.copy_parameters(self)
                self._initialized = True
            return self._instance.get_action(state)

    def __init__(self, algorithm, num_instances):
        self._main_instance = self.MainInstance(algorithm)
        self._async_instances = [self.AsyncInstance(algorithm, i) for i in range(num_instances)]
        for async_instance in self._async_instances:
            self._main_instance.register_instance(async_instance)
        self._initial_equalization_done = False

    def update(self, instance_id, state, action, reward, next_state, is_terminal):
        self._async_instances[instance_id].update(self._main_instance, state, action, reward, next_state, is_terminal)

    def get_action(self, instance_id, state):
        return self._async_instances[instance_id].get_action(self._main_instance, state)


class A3C(AsyncAlgorithm, pg.AdvantageActorCriticBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_sample(self, state, action, next_state, reward, is_terminal):
        self._observed_states.append(state)
        self._observed_actions.append(action)
        self._observed_next_states.append(next_state)
        self._observed_rewards.append(reward)
        self._observed_target_factors.append(0 if is_terminal else 1)

    def needs_update(self, is_terminal):
        self._steps_since_update += 1
        if is_terminal or self._steps_since_update >= self._steps_per_update:
            self._steps_since_update = 0
            return True
        return False

    def get_update_gradients(self):
        return self._actor_gradients, self._critic_gradients

    def get_update_feed_dict(self):
        feed_dict = {self._state: self._observed_states,
                     self._action: self._observed_actions,
                     self._next_state: self._observed_next_states,
                     self._reward: self._observed_rewards,
                     self._target_factor: self._observed_target_factors}
        self._empty_observervations()
        return feed_dict

    def update_with_gradients_ops(self, gradients):
        return [self._actor_optimizer.apply_gradients(gradients[0]),
                self._critic_optimizer.apply_gradients(gradients[1])] + \
               self._td_learner.copy_weights_ops
