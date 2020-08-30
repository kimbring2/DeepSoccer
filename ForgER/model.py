import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import tensorflow.keras.backend as K
from tensorflow.python.keras.engine.base_layer import InputSpec
mapping = dict()


def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk


def get_network_builder(name):
    """
    If you want to register your own network outside models.py, you just need:

    Usage Example:
    -------------
    from ForgER.model import register
    @register("your_network_name")
    def your_network_define(**net_kwargs):
        ...
        return network_fn

    """
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError('Registered networks:', ', '.join(mapping.keys()))


class DuelingModel(tf.keras.Model):
    def __init__(self, units, action_dim, reg=1e-6, noisy=True):
        super(DuelingModel, self).__init__()
        reg = {'kernel_regularizer': l2(reg), 'bias_regularizer': l2(reg)}
        if noisy:
            layer = NoisyDense
        else:
            layer = Dense
            
        kernel_init = tf.keras.initializers.VarianceScaling(scale=2.)
        self.h_layers = Sequential([layer(num, 'relu', use_bias=True, kernel_initializer=kernel_init,
                                          **reg) for num in units[:-1]])
        self.a_head = layer(units[-1]/2, 'relu', use_bias=True, kernel_initializer=kernel_init, **reg)
        self.v_head = layer(units[-1]/2, 'relu', use_bias=True, kernel_initializer=kernel_init, **reg)
        self.a_head1 = layer(action_dim, use_bias=True, kernel_initializer=kernel_init, **reg)
        self.v_head1 = layer(1, use_bias=True, kernel_initializer=kernel_init, **reg)

    @tf.function
    def call(self, inputs):
        print('Building model')
        features = self.h_layers(inputs)
        advantage, value = self.a_head(features), self.v_head(features)
        advantage, value = self.a_head1(advantage), self.v_head1(value)
        advantage = advantage - tf.reduce_mean(advantage, axis=-1, keepdims=True)
        out = value + advantage
        return out


class ClassicCnn(tf.keras.Model):
    def __init__(self, filters, kernels, strides, activation='relu', reg=1e-6):
        super(ClassicCnn, self).__init__()
        reg = l2(reg)
        kernel_init = tf.keras.initializers.VarianceScaling(scale=2.)
        self.cnn = Sequential(Conv2D(filters[0], kernels[0], strides[0], activation=activation,
                                     kernel_regularizer=reg, kernel_initializer=kernel_init), name='CNN')
        for f, k, s in zip(filters[1:], kernels[1:], strides[1:]):
            self.cnn.add(Conv2D(f, k, s, activation=activation, kernel_regularizer=reg,
                                kernel_initializer=kernel_init))
        self.cnn.add(Flatten())

    @tf.function
    def call(self, inputs):
        return self.cnn(inputs)


class MLP(tf.keras.Model):
    def __init__(self, units, activation='relu', reg=1e-6):
        super(MLP, self).__init__()
        reg = l2(reg)
        self.model = Sequential([Dense(num, activation, kernel_regularizer=reg, bias_regularizer=reg)
                                 for num in units])

    @tf.function
    def call(self, inputs):
        return self.model(inputs)


class NoisyDense(Dense):
    # factorized noise
    def __init__(self, units, *args, **kwargs):
        self.output_dim = units
        self.f = lambda x: tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))
        super(NoisyDense, self).__init__(units, *args, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=None)

        self.kernel_sigma = self.add_weight(shape=(self.input_dim, self.units),
                                            initializer=self.kernel_initializer,
                                            name='sigma_kernel',
                                            regularizer=self.kernel_regularizer,
                                            constraint=None)

        if self.use_bias:
            self.bias = self.add_weight(shape=(1, self.units),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=None)

            self.bias_sigma = self.add_weight(shape=(1, self.units,),
                                              initializer=self.bias_initializer,
                                              name='bias_sigma',
                                              regularizer=self.bias_regularizer,
                                              constraint=None)
        else:
            self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.input_dim})
        self.built = True

    def call(self, inputs):
        if inputs.shape[0]:
            kernel_input = self.f(tf.random.normal(shape=(inputs.shape[0], self.input_dim, 1)))
            kernel_output = self.f(tf.random.normal(shape=(inputs.shape[0], 1, self.units)))
        else:
            kernel_input = self.f(tf.random.normal(shape=(self.input_dim, 1)))
            kernel_output = self.f(tf.random.normal(shape=(1, self.units)))
        kernel_epsilon = tf.matmul(kernel_input, kernel_output)

        w = self.kernel + self.kernel_sigma * kernel_epsilon

        output = tf.matmul(tf.expand_dims(inputs, axis=1), w)

        if self.use_bias:
            b = self.bias + self.bias_sigma * kernel_output
            output = output + b
        if self.activation is not None:
            output = self.activation(output)
        output = tf.squeeze(output, axis=1)
        return output


@register("deepsoccer_dqfd")
def make_model(name, obs_space, action_space, reg=1e-5):
    pov = tf.keras.Input(shape=(128, 128, 5))
    normalized_pov = pov / 255
    pov_base = ClassicCnn([32, 128, 128], [8, 4, 3], [4, 2, 1], reg=reg)(normalized_pov)
    head = DuelingModel([1024], action_space.n, reg=reg)(pov_base)
    model = tf.keras.Model(inputs={'pov': pov}, outputs=head, name=name)
    return model


@register("flat_dqfd")
def make_model(name, obs_space, action_space, reg=1e-5):
    features = tf.keras.Input(shape=obs_space.shape)
    feat_base = MLP([64,64], activation='tanh', reg=reg)(features)
    head = DuelingModel([512], action_space.n, reg=reg, noisy=False)(feat_base)
    model = tf.keras.Model(inputs={'features': features}, outputs=head, name=name)
    return model

