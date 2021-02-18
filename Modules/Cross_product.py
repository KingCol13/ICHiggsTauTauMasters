# coding: utf-8

"""
Based TensorFlow implementation of the Lorentz Boost Network (LBN). https://arxiv.org/abs/1812.09722.
Modified by Alie to only include the cross product
"""


__author__ = "Marcel Rieger"
__copyright__ = "Copyright 2018-2020, Marcel Rieger"
__license__ = "BSD"
__credits__ = ["Martin Erdmann", "Erik Geiser", "Yannik Rath", "Marcel Rieger"]
__contact__ = "https://github.com/riga/LBN"
__email__ = "marcel.rieger@cern.ch"
__version__ = "1.2.1"

__all__ = ["LBN", "LBNLayer", "FeatureFactoryBase", "FeatureFactory"]


import functools

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops


# tf version flag
TF2 = tf.__version__.startswith("2.")


class LBN(object):
    """
    Lorentz Boost Network (LBN) class.

    Usage example:

    .. code-block:: python

        # initialize the LBN
        lbn = LBN(10, boost_mode=LBN.PAIRS)

        # create a feature tensor based on input four-vectors
        features = lbn(four_vectors)

        # use the features as input for a subsequent, application-specific network
        ...

    *n_particles* and *n_restframes* are the number of particle and rest-frame combinations to
    build. Their interpretation depends on the *boost_mode*. *n_restframes* is only used for the
    *PRODUCT* mode. It is inferred from *n_particles* for *PAIRS* and *COMBINATIONS*.

    *epsilon* is supposed to be a small number that is used in various places for numerical
    stability. When not *None*, *seed* is used to seed random number generation for trainable
    weights. *trainable* is passed to *tf.Variable* during weight generation. *name* is the main
    namespace of the LBN and defaults to the class name.

    *feature_factory* must be a subclass of :py:class:`FeatureFactoryBase` and provides the
    available, generic mappings from boosted particles to output features of the LBN. If *None*, the
    default :py:class:`FeatureFactory` is used.

    *particle_weights* and *restframe_weights* can refer to externally defined variables with custom
    initialized weights. If set, their shape must match the number of combinations and inputs. For
    simple initialization tests, *weight_init* can be a tuple containing the Gaussian mean and
    standard deviation that is passed to ``tf.random.normal``. When *None*, and the weight tensors
    are created internally, mean and standard deviation default to *0* and *1 / combinations*. When
    *abs_particle_weights* (*abs_restframe_weights*) is *True*, ``tf.abs`` is applied to the
    particle (rest frame) weights. When *clip_particle_weights* (*clip_restframe_weights*) is
    *True*, particle (rest frame) weights are clipped at *epsilon*, or at the passed value if it is
    not a boolean. Note that the abs operation is applied before clipping.

    When the number of features per input particle is larger than four, the subsequent values are
    interpreted as auxiliary features. Similar to the combined particles and restframes, these
    features are subject to linear combinations to create new, embedded representations. The number
    number of combinations, *n_auxiliaries*, defaults to the number of boosted output particles.
    Their features are concatenated to the vector of output features. The weight tensor
    *aux_weights* is used to create the combined feautres. When given, it should have the shape
    ``(n_in * (n_dim - 4)) x n_auxiliaries``.

    Instances of this class store most of the intermediate tensors (such as inputs, combinations
    weights, boosted particles, boost matrices, raw features, etc) for later inspection. Note that
    most of these tensors are set after :py:meth:`build` (or the :py:meth:`__call__` shorthand as
    shown above) are invoked.
    """

    def __init__(self, n_particles, n_restframes=None, n_auxiliaries=None, 
            feature_factory=None, particle_weights=None, abs_particle_weights=True,
            clip_particle_weights=False, restframe_weights=None, abs_restframe_weights=True,
            clip_restframe_weights=False, aux_weights=None, weight_init=None, epsilon=1e-5,
            seed=None, trainable=True, name=None):
        
        #boost_mode=PAIRS, #this needs to ber removed from the input list or Pair needs to be included above
        
        super(LBN, self).__init__()
        self.n_out = n_particles 

        # auxiliary weights
        self.aux_weights = aux_weights

        # custom weight init parameters in a tuple (mean, stddev)
        self.weight_init = weight_init

        # epsilon for numerical stability
        self.epsilon = epsilon

        # random seed
        self.seed = seed

        # trainable flag
        self.trainable = trainable

        # internal name
        self.name = name or self.__class__.__name__

        # sizes that are set during build
        self.n_in = None  # number of input particles
        self.n_dim = None  # size per input vector, must be four or higher
        self.n_aux = None  # size of auxiliary features per input vector (n_dim - 4)

        # constants
        self.I = None  # the I matrix
        self.U = None  # the U matrix

        # tensor of input vectors
        self.inputs = None

        # split input tensors
        self.inputs_E = None  # energy column of inputs
        self.inputs_px = None  # px column of inputs
        self.inputs_py = None  # py column of inputs
        self.inputs_pz = None  # pz column of inputs
        self.inputs_aux = None  # auxiliary columns of inputs

        # features
        self.n_features = None  # total number of produced features
        self.boosted_features = None  # features of boosted particles
        #I want now the boosted features to be the normal features
        self.aux_features = None  # auxiliary features (batch, n_in * n_aux, n_auxiliaries)
        self.features = None  # final, combined output features

        # initialize the feature factory
        if feature_factory is None:
            feature_factory = FeatureFactory
        elif not issubclass(feature_factory, FeatureFactoryBase):
            raise TypeError("feature_factory '{}' is not a subclass of FeatureFactoryBase".format(
                feature_factory))
        self.feature_factory = feature_factory(self)

        # the function that either builds the graph lazily, or can be used as an eager callable
        self._op = None

    @property
    def built(self):
        return self._op is not None

    @property
    def available_features(self):
        """
        Shorthand to access the list of available features in the :py:attr:`feature_factory`.
        """
        return list(self.feature_factory._feature_funcs.keys())

    def __call__(self, inputs, **kwargs):
        """
        Returns the LBN output features for specific *inputs*. It is ensured that the graph or eager
        callable are lazily created the first time this method is called by forwarding both *inputs*
        and *kwargs* to :py:meth:`build`.
        """
        # make sure the lbn op is built
        if not self.built:
            self.build(inputs.shape, **kwargs)

        # invoke it
        return self._op(inputs)

    def build(self, input_shape, features=("E", "px", "py", "pz"), external_features=None):
        """
        Builds the LBN structure layer by layer within dedicated variable scopes. *input_shape* must
        be a list, tuple or TensorShape object describing the dimensions of the input four-vectors.
        *features* and *external_features* are forwarded to :py:meth:`build_features`.
        """
        with tf.name_scope(self.name):
            # store shape and size information
            self.infer_sizes(input_shape)

            # setup variables
            with tf.name_scope("variables"):
                #self.setup_weight("particle", (self.n_in, self.n_particles), 1)

                #if self.boost_mode != self.COMBINATIONS:
                    #self.setup_weight("restframe", (self.n_in, self.n_restframes), 2)

                if self.n_aux > 0:
                    self.setup_weight("aux", (self.n_in, self.n_auxiliaries, self.n_aux), 3)

            # constants
            with tf.name_scope("constants"):
                self.build_constants()

        # compute the number of total features
        self.n_features = 0
        # lbn features
        for feature in features:
            self.n_features += self.feature_factory._feature_funcs[feature]._shape_func(self.n_out)
        # auxiliary features
        if self.n_aux > 0:
            self.n_features += self.n_out * self.n_aux
        # external features
        if external_features is not None:
            self.n_features += external_features.shape[1]

        # also store the op that can be used to either create a graph or an eager callable
        def op(inputs):
            with tf.name_scope(self.name):
                with tf.name_scope("inputs"):
                    self.handle_input(inputs)

                with tf.name_scope("features"):
                    if self.n_aux > 0:
                        with tf.name_scope("auxiliary"):
                            self.build_auxiliary()

                    self.build_features(features=features, external_features=external_features)

            return self.features

        self._op = op

    def infer_sizes(self, input_shape):
        """
        Infers sizes based on the shape of the input tensor.
        """
        if not isinstance(input_shape, (tuple, list, tf.TensorShape)):
            input_shape = input_shape.shape

        self.n_in = int(input_shape[-2])
        self.n_dim = int(input_shape[-1])

        if self.n_dim < 4:
            raise Exception("input dimension must be at least 4")
        self.n_aux = self.n_dim - 4

    def setup_weight(self, prefix, shape, seed_offset=0):
        """
        Sets up the variable tensors representing linear coefficients for the combinations of
        particles and rest frames. *prefix* must either be ``"particle"``, ``"restframe"`` or
        ``"aux"``. *shape* describes the shape of the weight variable to create. When not *None*,
        the seed attribute of this instance is incremented by *seed_offset* and passed to the
        variable constructor.
        """
        if prefix not in ["particle", "restframe", "aux"]:
            raise ValueError("unknown prefix '{}'".format(prefix))

        # define the weight name
        name = "{}_weights".format(prefix)

        # when the variable is already set, i.e. passed externally, validate the shape
        # otherwise, create a new variable
        W = getattr(self, name, None)
        if W is not None:
            # verify the shape
            w_shape = tuple(W.shape.as_list())
            if w_shape != shape:
                raise ValueError("the shape of variable {} {} does not match {}".format(
                    name, shape, w_shape))
        else:
            # define mean and stddev of weight init
            if isinstance(self.weight_init, tuple):
                mean, stddev = self.weight_init
            else:
                mean, stddev = 0., 1. / shape[1]

            # apply the seed offset when not None
            seed = (self.seed + seed_offset) if self.seed is not None else None

            # create and save the variable
            W = tf.Variable(tf.random.normal(shape, mean, stddev, dtype=tf.float32,
                seed=seed), name=name, trainable=self.trainable)
            setattr(self, name, W)

    def build_constants(self):
        """
        Builds the internal constants for the boost matrix.
        """
        # 4x4 identity
        self.I = tf.constant(np.identity(4), tf.float32)

        # U matrix
        self.U = tf.constant([[-1, 0, 0, 0]] + 3 * [[0, -1, -1, -1]], tf.float32)

    def handle_input(self, inputs):
        """
        Takes the passed *inputs* and stores internal tensors for further processing and later
        inspection.
        """
        # store the input vectors
        self.inputs = inputs

        # also store the four-vector components maybe not right, need to print them
        self.inputs_E = self.inputs[..., 0]
        self.inputs_px = self.inputs[..., 1]
        self.inputs_py = self.inputs[..., 2]
        self.inputs_pz = self.inputs[..., 3] 
        #it used to be [..., 3] but working better that way, copied from alie_testing.py
        #added by Alie
        self.inputs_pvect = self.inputs[..., 1:]
        
        print(self.inputs_pvect, 'This is the pvect total')

        # split auxiliary inputs
        if self.n_aux > 0:
            self.inputs_aux = self.inputs[..., 4:]


    def build_auxiliary(self):
        """
        Build combinations of auxiliary input features using the same approach as for particles and
        restframes.
        """
        if self.n_aux <= 0:
            raise Exception("cannot build auxiliary features when n_aux is not positive")

        # build the features via a simple matmul, mapped over the last axis
        self.aux_features = tf.concat([
            tf.matmul(self.inputs_aux[..., i], self.aux_weights[..., i]) 
            for i in range(self.n_aux)
        ], axis=1)

    def build_features(self, features=("E", "px", "py", "pz"), external_features=None):
        """
        Builds the output features. *features* should be a list of feature names as registered to
        the :py:attr:`feature_factory` instance. When *None*, the default features
        ``["E", "px", "py", "pz"]`` are built. *external_features* can be a list of tensors of
        externally produced features, that are concatenated with the built features.
        """
        symbolic = _is_symbolic(self.inputs)  #check if this line is causing any trouble

        # clear the feature caches
        self.feature_factory.clear_caches()

        # create the list of feature ops to concat
        concat = []
        for name in features:
            func = getattr(self.feature_factory, name)
            if func is None:
                raise ValueError("unknown feature '{}'".format(name))
            concat.append(func(_symbolic=symbolic))

        # save intermediate boosted features
        #In our case they won;t bee boosted features but normal ones
        self.boosted_features = tf.concat(concat, axis=-1)

        # add auxiliary features
        if self.n_aux > 0:
            concat.append(self.aux_features)

        # add external features
        if external_features is not None:
            if isinstance(external_features, (list, tuple)):
                concat.extend(list(external_features))
            else:
                concat.append(external_features)

        # save combined features
        self.features = tf.concat(concat, axis=-1)


def _is_symbolic(t):
    """
    Returs *True* when a tensor *t* is a symbolic tensor.
    """
    if len(t.shape) > 0 and t.shape[0] is None:
        return True
    elif callable(getattr(tf_ops, "_is_keras_symbolic_tensor", None)) and \
            tf_ops._is_keras_symbolic_tensor(t):
        return True
    elif getattr(tf_ops, "EagerTensor", None) is not None and isinstance(t, tf_ops.EagerTensor):
        return False
    elif callable(getattr(t, "numpy", None)):
        return False
    else:
        # no other check to perform, assume it is eager
        return False


class FeatureFactoryBase(object):
    """
    Base class of the feature factory. It does not implement actual features but rather the
    feature wrapping and tensor caching functionality. So-called hidden features are also subject to
    caching but are not supposed to be accessed by the LBN. They rather provide intermediate results
    that are used in multiple places and retained for performance purposes.
    """

    DISABLE_CACHE = False

    @classmethod
    def feature(cls, shape_func, hidden=False):
        def decorator(func):
            name = func.__name__

            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                no_cache = kwargs.get("_no_cache", self.DISABLE_CACHE)
                symbolic = kwargs.get("_symbolic", False)

                # get the result of the wrapped feature, with or without caching
                if no_cache:
                    return tf.identity(func(self, *args, **kwargs), name=name)
                else:
                    cache = self._symbolic_tensor_cache if symbolic else self._eager_tensor_cache
                    if name not in cache:
                        cache[name] = tf.identity(func(self, *args, **kwargs), name=name)
                    return cache[name]

            # store attributes on the feature wrapper for later use
            wrapper._feature = True
            wrapper._func = func
            wrapper._shape_func = shape_func
            wrapper._hidden = hidden

            return wrapper

        return decorator

    @classmethod
    def hidden_feature(cls, func):
        return cls.feature(None, hidden=True)(func)

    @classmethod
    def single_feature(cls, func):
        shape_func = lambda n_out: n_out
        return cls.feature(shape_func)(func)

    @classmethod
    def pair_feature(cls, func):
        shape_func = lambda n_out: (n_out**2 - n_out) / 2
        return cls.feature(shape_func)(func)

    def __init__(self, lbn):
        super(FeatureFactoryBase, self).__init__()

        # reference to the lbn instance
        self.lbn = lbn

        # some shorthands
        self.n = lbn.n_out
        self.epsilon = lbn.epsilon

        # cached symbolic tensors stored by name
        self._symbolic_tensor_cache = {}

        # cached eager tensors stored by name
        self._eager_tensor_cache = {}

        # dict of registered feature functions without hidden ones
        self._feature_funcs = {}
        for attr in dir(self):
            func = getattr(self, attr)
            if getattr(func, "_feature", False) and not func._hidden:
                self._feature_funcs[attr] = func

    def clear_symbolic_cache(self):
        """
        Clears the current eager tensor cache.
        """
        self._symbolic_tensor_cache.clear()

    def clear_eager_cache(self):
        """
        Clears the current eager tensor cache.
        """
        self._eager_tensor_cache.clear()

    def clear_caches(self):
        """
        Clears both the current eager and symbolic tensor caches.
        """
        self.clear_symbolic_cache()
        self.clear_eager_cache()


class FeatureFactory(FeatureFactoryBase):
    """
    Default feature factory implementing various generic feature mappings.
    """

    def __init__(self, lbn):
        super(FeatureFactory, self).__init__(lbn)
        # pairwise features are computed by multiplying row and column vectors to obtain a
        # matrix from which we want to extract the values of the upper triangle w/o diagonal,
        # so store these upper triangle indices for later use in tf.gather
        self.triu_indices = triu_range(self.n)
        #self.tril_indices = tril_range(self.n)  #added by alie, useful for cross product method
 
    @FeatureFactoryBase.single_feature
    def E(self, **opts):
        """
        Energy.
        """
        #E=self.lbn.boosted_particles[..., 0]
        
        E=self.lbn.inputs_E #for cross product we only use the inputs
        print("\n E shape \n",E.shape)
        return E

    @FeatureFactoryBase.single_feature
    def px(self, **opts):
        """
        Momentum component x.
        """
        
#         self.lbn.boosted_particles[..., 1]
        return self.lbn.inputs_px

    @FeatureFactoryBase.single_feature
    def py(self, **opts):
        """
        Momentum component y.
        """
#         self.lbn.boosted_particles[..., 2]
        return self.lbn.inputs_py

    @FeatureFactoryBase.single_feature
    def pz(self, **opts):
        """
        Momentum component z.
        """
#         self.lbn.boosted_particles[..., 3]
        return self.lbn.inputs_pz

    @FeatureFactoryBase.hidden_feature
    def _pvec(self, **opts):
        """
        Momentum vector. Hidden.
        """
#         self.lbn.boosted_particles[..., 1:]

        print(self.lbn.inputs_pvect[1], 'This is pvect, it should have shape ')
        return self.lbn.inputs_pvect

    
    #ADDED BY ALIE ! #next: include negative part ?
    @FeatureFactoryBase.pair_feature
    def cross_product_z(self, **opts):
        """
        Z component of cross product between momenta of each pair of particles
        """
        
        #we need to expand in 2d to have the right matrix arrangement
        cross_z = tf.expand_dims(self.px(**opts), axis=-1)*tf.expand_dims(self.py(**opts), axis=-2)

        #only transpose the two last sides, we want some pairwise operations
        cross_z_T=tf.einsum('aij -> aji', cross_z) 
        
        #and then substract and keep the right half of the triangle to have cross product
        yy = tf.gather(tf.reshape(cross_z-cross_z_T, [-1, self.n**2]), self.triu_indices, axis=1)
        return yy
    
    #ADDED BY ALIE ! #next: make it actually clean and not repeat the same thing 3 times ?
    @FeatureFactoryBase.pair_feature
    def cross_product_x(self, **opts):
        """
        X component of cross product between momenta of each pair of particles
        """
        #we need to expand in 2d to have the right matrix arrangement
        cross_x = tf.expand_dims(self.py(**opts), axis=-1)*tf.expand_dims(self.pz(**opts), axis=-2)

        #only transpose the two last sides, we want some pairwise operations
        cross_x_T=tf.einsum('aij -> aji', cross_x) 
        
        #and then substract and keep the right half of the triangle to have cross product
        yy = tf.gather(tf.reshape(cross_x-cross_x_T, [-1, self.n**2]), self.triu_indices, axis=1)
        return yy
    
    
     #ADDED BY ALIE !
    @FeatureFactoryBase.pair_feature
    def cross_product_y(self, **opts):
        """
        X component of cross product between momenta of each pair of particles
        """
        #we need to expand in 2d to have the right matrix arrangement
        cross_y = tf.expand_dims(self.pz(**opts), axis=-1)*tf.expand_dims(self.px(**opts), axis=-2)

        #only transpose the two last sides, we want some pairwise operations
        cross_y_T=tf.einsum('aij -> aji', cross_y) 
        
        #and then substract and keep the right half of the triangle to have cross product
        yy = tf.gather(tf.reshape(cross_y-cross_y_T, [-1, self.n**2]), self.triu_indices, axis=1)
        return yy

def tril_range(n, k=-1):
    """
    Returns a 1D numpy array containing all lower triangle indices of a square matrix with size *n*.
    *k* is the offset from the diagonal.
    """
    tril_indices = np.tril_indices(n, k)
    return np.arange(n**2).reshape(n, n)[tril_indices]


def triu_range(n, k=1):
    """
    Returns a 1D numpy array containing all upper triangle indices of a square matrix with size *n*.
    *k* is the offset from the diagonal.
    """
    triu_indices = np.triu_indices(n, k)
    return np.arange(n**2).reshape(n, n)[triu_indices]


class LBNLayer(tf.keras.layers.Layer):
    """
    Keras layer of the :py:class:`LBN` that forwards the standard interface of :py:meth:`__init__`
    and py:meth:`__call__`.

    .. py:attribute:: lbn
       type: LBN

       Reference to the internal :py:class:`LBN` instance that is initialized with the contructor
       arguments of this class.
    """

    def __init__(self, input_shape, *args, **kwargs):
        # store and remove kwargs that are not passed to the LBN but to the layer init
        layer_kwargs = {
            "input_shape": input_shape,
            "dtype": kwargs.pop("dtype", None),
            "dynamic": kwargs.pop("dynamic", False),
        }
        # for whatever reason, keras calls this contructor again
        # with batch_input_shape set when input_shape was accepted
        if "batch_input_shape" in kwargs:
            layer_kwargs["batch_input_shape"] = kwargs.pop("batch_input_shape")

        # store names of features to build
        self._features = kwargs.pop("features", None)


        # store external features to concatenate with the lbn outputs
        self._external_features = kwargs.pop("external_features", None)

        # create the LBN instance with the remaining arguments
        self.lbn = LBN(*args, **kwargs)

        # the input_shape is mandatory so we can build right away
        self.build(input_shape)

        # layer init
        super(LBNLayer, self).__init__(name=self.lbn.name, trainable=self.lbn.trainable,
            **layer_kwargs)

    def build(self, input_shape):
        # build the lbn
        self.lbn.build(input_shape, features=self._features,
            external_features=self._external_features)

        # store references to the trainable weights
        # (not necessarily the weights used in combinations)
        self.aux_weights = self.lbn.aux_weights
        super(LBNLayer, self).build(input_shape)

    def call(self, inputs):
        return self.lbn(inputs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.lbn.n_features)

    def get_config(self):
        config = super(LBNLayer, self).get_config()
        config.update({
            "input_shape": (self.lbn.n_in, self.lbn.n_dim),
            "n_auxiliaries": self.lbn.n_auxiliaries,
            "epsilon": self.lbn.epsilon,
            "seed": self.lbn.seed,
            "features": self._features,
        })
        return config
