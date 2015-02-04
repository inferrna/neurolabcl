# -*- coding: utf-8 -*-
"""
Functions of initialization  layers

"""


from neurolab import mynp  as np


def init_rand(layer, min=-0.5, max=0.5, init_prop='w'):
    """
    Initialize the specified property of the layer
    random numbers within specified limits

    :Parameters:
        layer:
            Initialized layer
        min: float (default -0.5)
            minimum value after the initialization
        max: float (default 0.5)
            maximum value after the initialization
        init_prop: str (default 'w')
            name of initialized property, must be in layer.np

    """

    if init_prop not in layer.np:
        raise ValueError('Layer not have attibute "' + init_prop + '"')
    layer.np[init_prop] = np.random.uniform(min, max, layer.np[init_prop].shape)

def initwb_reg(layer):
    """
    Initialize weights and bias
    in the range defined by the activation function (transf.inp_active)

    """
    active = layer.transf.inp_active[:]

    if np.isinf(active[0]):
        active[0] = -100.0

    if np.isinf(active[1]):
        active[1] = 100.0

    min = active[0] / (2 * layer.cn)
    max = active[1] / (2 * layer.cn)

    init_rand(layer, min, max, 'w')
    if 'b' in layer.np:
        init_rand(layer, min, max, 'b')


class InitRand:
    """
    Initialize the specified properties of the layer
    random numbers within specified limits

    """
    def __init__(self, minmax, init_prop):
        """
        :Parameters:
            minmax: list of float
                [min, max] init range
            init_prop: list of dicts
                names of initialized propertis. Example ['w', 'b']

        """
        self.min = minmax[0]
        self.max = minmax[1]
        self.properties = init_prop

    def __call__(self, layer):
        for property in self.properties:
            init_rand(layer, self.min, self.max, property)
        return


def init_zeros(layer):
    """
    Set all layer properties of zero

    """
    for k in layer.np:
        layer.np[k].fill(0.0)
    return


def midpoint(layer):
    """
    Sets weight to the center of the input ranges

    """
    mid = layer.inp_minmax.mean(axis=1)
    for i, w in enumerate(layer.np['w']):
        layer.np['w'][i] = mid.copy()
    return

def initnw(layer):
    """
    Nguyen-Widrow initialization function

    """
    ci = layer.ci
    cn = layer.cn
    w_fix = 0.7 * cn ** (1. / ci)
    w_rand = np.random.rand(cn, ci) * 2 - 1
    print("0type(w_rand) == ", type(w_rand))
    print("0.1type(w_rand) == ", type(np.square(w_rand)))
    # Normalize
    if ci == 1:
        w_rand = w_rand / np.abs(w_rand)
    else:
        print("w_rand.shape == ", w_rand.shape)
        print("type(np.sqrt(1. / np.square(w_rand))) == ",
               type(np.sqrt(1. / np.square(w_rand))))
        print("type(np.sqrt(1. / np.square(w_rand).sum(axis=1))) == ",
               type(np.sqrt(1. / np.square(w_rand).sum(axis=1))))
        print("np.sqrt(1. / np.square(w_rand).sum(axis=1).reshape(cn, 1)).shape == ",
               np.sqrt(1. / np.square(w_rand).sum(axis=1).reshape(cn, 1)).shape)
        w_rand = w_rand * np.sqrt(1. / np.square(w_rand).sum(axis=1).reshape(cn, 1))

    print("1type(w_rand) == ", type(w_rand))
    print("0type(w_fix) == ", type(w_fix))
    w = w_fix * w_rand
    print("0type(w) == ", type(w))
    #print("0shape w_fix == ", w_fix.shape)
    print("0shape np.linspace(-1, 1, cn) == ", np.linspace(-1, 1, cn).shape)
    print("0shape np.sign(w[:, 0]) == ", np.sign(w[:, 0]).shape)
    b = np.array([0]) if cn == 1 else w_fix * np.linspace(-1, 1, cn) * np.sign(w[:, 0])

    # Scaleble to inp_active
    amin, amax  = layer.transf.inp_active
    amin = -1 if amin == -np.Inf else amin
    amax = 1 if amax == np.Inf else amax

    x = 0.5 * (amax - amin)
    y = 0.5 * (amax + amin)
    print("1type(w) == ", type(w))
    w = x * w
    print("2type(w) == ", type(w))
    b = x * b + y

    # Scaleble to inp_minmax
    minmax = layer.inp_minmax.copy()
    print(type(minmax))
    print(type(np.isneginf(minmax)))
    print(minmax)
    print(minmax.shape)
    print(np.isneginf(minmax))
    print(minmax[np.isneginf(minmax)])
    minmax[np.isneginf(minmax)] = -1
    minmax[np.isinf(minmax)] = 1

    x = 2. / (minmax[:, 1] - minmax[:, 0])
    print("x.shape == ", x.shape)
    print("type(x) == ", type(x))
    print("w.shape == ", w.shape)
    print("3type(w) == ", type(w))
    y = 1. - minmax[:, 1] * x
    w = w * x
    print("b.shape == ", b.shape)
    print("dot.shape == ", np.dot(w, y).shape)

    b = b + np.dot(w, y)

    print("w.shape == ", w.shape)
    layer.np['w'][:] = w
    print("b.shape == ", b.shape)
    print("layer.np['b'].shape == ", layer.np['b'][:].shape)
    layer.np['b'][:] = b

    return
