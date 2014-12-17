# -*- coding: utf-8 -*-
"""
Train algorithms based  gradients algorihms
===========================================

.. autofunction:: train_gd
.. autofunction:: train_gdm
.. autofunction:: train_gda
.. autofunction:: train_gdx
.. autofunction:: train_rprop

Train algorithms based on Winner Take All - rule
================================================
.. autofunction:: train_wta
.. autofunction:: train_cwta

Train algorithms based on spipy.optimize
========================================
.. autofunction:: train_bfgs
.. autofunction:: train_cg
.. autofunction:: train_ncg

Train algorithms for LVQ networks
=================================
.. autofunction:: train_lvq

Delta rule
==========

.. autofunction:: train_delta

"""

import functools
from neurolab import mynp  as np
from neurolab.core import Train
import neurolab.tool as tool


######<<<delta
"""
Train algorithm based on Delta - rule

"""
class TrainDelta(Train):
    """ 
    Train with Delta rule
    
    :Support networks:
        newp (one-layer perceptron)
    :Parameters:
        input: array like (l x net.ci)
            train input patterns
        epochs: int (default 500)
            Number of train epochs
        show: int (default 100)
            Print period
        goal: float (default 0.01)
            The goal of train
        lr: float (default 0.01)
            learning rate
    
    """
    
    def __init__(self, net, input, target, lr=0.01):
        self.lr = lr
        
    def __call__(self, net, input, target):
        layer = net.layers[0]
        while True:
            e = self.error(net, input, target)
            self.epochf(e, net, input, target)
            for inp, tar in zip(input, target):
                out = net.step(inp)
                err = tar - out
                err.shape =  err.size, 1
                inp.shape = 1, inp.size
                layer.np['w'] += self.lr * err * inp
                err.shape =  err.size
                layer.np['b'] += self.lr * err
        return None
######delta>>>

######<<<lvq
"""
Train algorithms for LVQ networks

"""
class TrainLVQ(Train):
    """
    LVQ1 train function
    
    :Support networks:
        newlvq
    :Parameters:
        input: array like (l x net.ci)
            train input patterns
        epochs: int (default 500)
            Number of train epochs
        show: int (default 100)
            Print period
        goal: float (default 0.01)
            The goal of train
        lr: float (defaults 0.01)
            learning rate
        adapt bool (default False)
            type of learning
    
    """
    
    def __init__(self, net, input, target, lr=0.01, adapt=True):
        self.adapt = adapt
        self.lr = lr
    
    def __call__(self, net, input, target):
        layer = net.layers[0]
        if self.adapt:
            while True:
                self.epochf(None, net, input, target)
                
                for inp, tar in zip(input, target):
                    out = net.step(inp)
                    err = tar - out
                    win = np.argmax(layer.out)
                    if np.max(err) == 0.0:
                        layer.np['w'][win] += self.lr * (inp - layer.np['w'][win])
                    else:
                        layer.np['w'][win] -= self.lr * (inp - layer.np['w'][win])
        else:
            while True:
                output = []
                winners = []
                for inp, tar in zip(input, target):
                    out = net.step(inp)
                    output.append(out)
                    winners.append(np.argmax(layer.out))
                
                e = self.error(net, input, target, output)
                self.epochf(e, net, input, target)
                
                error = target - output
                sign = np.sign((np.max(error, axis=1) == 0) - 0.5)
                layer.np['w'][winners] += self.lr * (input - layer.np['w'][winners])
        return None
######lvq>>>

######<<<wta
"""
Train algorithm based on Winner Take All - rule

"""
class TrainWTA(Train):
    """ 
    Winner Take All algorithm
    
    :Support networks:
        newc (Kohonen layer)
    :Parameters:
        input: array like (l x net.ci)
            train input patterns
        epochs: int (default 500)
            Number of train epochs
        show: int (default 100)
            Print period
        goal: float (default 0.01)
            The goal of train
    
    """
       
    def __init__(self, net, input, lr=0.01):
        # Init network!
        self.lr = lr
        for w in net.layers[0].np['w']:
            w[:] = input[np.random.randint(0, len(input))]

    def error(self, net, input):
        layer = net.layers[0]
        winner_output = np.zeros_like(input)
        output = net.sim(input)
        winners = np.argmax(output, axis=1)
        e =  layer.np['w'][winners] - input
        
        return net.errorf(e)
    
    def learn(self, net, input):
        layer = net.layers[0]

        for inp in input:
            out = net.step(inp)
            winner = np.argmax(out)
            d = layer.last_dist
            layer.np['w'][winner] += self.lr * d[winner] * (inp - layer.np['w'][winner])
        
        return None


class TrainCWTA(TrainWTA):
    """ 
    Conscience Winner Take All algorithm
    
    :Support networks:
        newc (Kohonen layer)
    :Parameters:
        input: array like (l x net.ci)
            train input patterns
        epochs: int (default 500)
            Number of train epochs
        show: int (default 100)
            Print period
        goal: float (default 0.01)
            The goal of train
    
    """
    
    def learn(self, net, input):
        layer = net.layers[0]

        for inp in input:
            out = net.step(inp)
            winner = np.argmax(out)
            d = layer.last_dist #TODO:^^_^^
            layer.np['conscience'][winner] += 1
            layer.np['w'][winner] += self.lr * d[winner] * (inp - layer.np['w'][winner])

        layer.np['conscience'].fill(1.0)
        return None
######wta>>>

######<<<spo
"""
Train algorithm based on spipy.optimize

"""
class TrainSO(Train):
    """
    Train class Based on scipy.optimize

    """

    def __init__(self, net, input, target, **kwargs):
        self.net = net
        self.input = input
        self.target = target
        self.kwargs = kwargs
        self.x = tool.np_get_ref(net)
        self.lerr = 1e10

    def grad(self, x):
        self.x[:] = x
        gr = tool.ff_grad(self.net, self.input, self.target)[1]
        return gr

    def fcn(self, x):
        self.x[:] = x
        err = self.error(self.net, self.input, self.target)
        self.lerr = err
        return err

    def step(self, x):
        self.epochf(self.lerr, self.net, self.input, self.target)

    def __call__(self, net, input, target):
        raise NotImplementedError("Call abstract metod __call__")


class TrainBFGS(TrainSO):
    """
    BroydenFletcherGoldfarbShanno (BFGS) method
    Using scipy.optimize.fmin_bfgs

    :Support networks:
        newff (multi-layers perceptron)
    :Parameters:
        input: array like (l x net.ci)
            train input patterns
        target: array like (l x net.co)
            train target patterns
        epochs: int (default 500)
            Number of train epochs
        show: int (default 100)
            Print period
        goal: float (default 0.01)
            The goal of train

    """

    def __call__(self, net, input, target):
        from scipy.optimize import fmin_bfgs
        if 'disp' not in self.kwargs:
            self.kwargs['disp'] = 0
        self.kwargs['maxiter'] = self.epochs

        x = fmin_bfgs(self.fcn, self.x.copy(), fprime=self.grad, callback=self.step,
                      **self.kwargs)
        self.x[:] = x


class TrainCG(TrainSO):
    """
    Newton-CG method
    Using scipy.optimize.fmin_ncg

    :Support networks:
        newff (multi-layers perceptron)
    :Parameters:
        input: array like (l x net.ci)
            train input patterns
        target: array like (l x net.co)
            train target patterns
        epochs: int (default 500)
            Number of train epochs
        show: int (default 100)
            Print period
        goal: float (default 0.01)
            The goal of train

    """

    def __call__(self, net, input, target):
        from scipy.optimize import fmin_cg
        if 'disp' not in self.kwargs:
            self.kwargs['disp'] = 0
        x = fmin_cg(self.fcn, self.x.copy(), fprime=self.grad, callback=self.step, **self.kwargs)
        self.x[:] = x
        return None

class TrainNCG(TrainSO):
    """
    Conjugate gradient algorithm
    Using scipy.optimize.fmin_ncg

    :Support networks:
        newff (multi-layers perceptron)
    :Parameters:
        input: array like (l x net.ci)
            train input patterns
        target: array like (l x net.co)
            train target patterns
        epochs: int (default 500)
            Number of train epochs
        show: int (default 100)
            Print period
        goal: float (default 0.01)
            The goal of train

    """

    def __call__(self, net, input, target):
        from scipy.optimize import fmin_ncg
        #if 'disp' not in self.kwargs:
        #    self.kwargs['disp'] = 0
        x = fmin_ncg(self.fcn, self.x.copy(), fprime=self.grad, callback=self.step, **self.kwargs)
        self.x[:] = x
        return None
######spo>>>

######<<<gd
"""
Train algorithm based  gradients algorithms

"""

class TrainGD(Train):
    """
    Gradient descent backpropagation
    
    :Support networks:
        newff (multi-layers perceptron)
    :Parameters:
        input: array like (l x net.ci)
            train input patterns
        target: array like (l x net.co)
            train target patterns
        epochs: int (default 500)
            Number of train epochs
        show: int (default 100)
            Print period
        goal: float (default 0.01)
            The goal of train
        lr: float (defaults 0.01)
            learning rate
        adapt bool (default False)
            type of learning
    """
    
    def __init__(self, net, input, target, lr=0.01, adapt=False):
        self.adapt = adapt
        self.lr = lr
        
    def __call__(self, net, input, target):
        if not self.adapt:
            while True:
                g, output = self.calc(net, input, target)
                e = self.error(net, input, target, output)
                self.epochf(e, net, input, target)
                self.learn(net, g)
        else:
            while True:
                for i in range(input.shape[0]):
                    g = self.calc(net, [input[i]], [target[i]])[0]
                    self.learn(net, g)
                e = self.error(net, input, target)
                self.epochf(e, net, input, target)
        return None
            
    def calc(self, net, input, target):
        g1, g2, output = tool.ff_grad(net, input, target)
        return g1, output
    
    def learn(self, net, grad):
        for ln, layer in enumerate(net.layers):
            layer.np['w'] -= self.lr * grad[ln]['w']
            layer.np['b'] -= self.lr * grad[ln]['b']
        return None
        

class TrainGD2(TrainGD):
    """
    Gradient descent backpropagation
    (another realization of TrainGD)
    
    :Support networks:
        newff (multi-layers perceptron)
    :Parameters:
        input: array like (l x net.ci)
            train input patterns
        target: array like (l x net.co)
            train target patterns
        epochs: int (default 500)
            Number of train epochs
        show: int (default 100)
            Print period
        goal: float (default 0.01)
            The goal of train
        lr: float (defaults 0.01)
            learning rate
        adapt bool (default False)
            type of learning
        
    """
    
    def __init__(self, net, input, target, lr=0.01, adapt=False):
        self.adapt = adapt
        self.lr = lr
        self.x = tool.np_get_ref(net)
    
    def calc(self, net, input, target):
        g1, g2, output = tool.ff_grad(net, input, target)
        return g2, output
    
    def learn(self, net, grad):
        self.x -= self.lr * grad

        
class TrainGDM(TrainGD):
    """
    Gradient descent with momentum backpropagation
    
    :Support networks:
        newff (multi-layers perceptron)
    :Parameters:
        input: array like (l x net.ci)
            train input patterns
        target: array like (l x net.co)
            train target patterns
        epochs: int (default 500)
            Number of train epochs
        show: int (default 100)
            Print period
        goal: float (default 0.01)
            The goal of train
        lr: float (defaults 0.01)
            learning rate
        adapt bool (default False)
            type of learning
    
    """

    def __init__(self, net, input, target, lr=0.01, adapt=False, mc=0.9):
        super(TrainGDM, self).__init__(net, input, target, lr, adapt)
        self.mc = mc
        self.dw = [0] * len(net.layers)
        self.db = [0] * len(net.layers)
    
    def learn(self, net, grad):
        #print 'GDM.learn'
        mc = self.mc
        lr = self.lr
        for ln, layer in enumerate(net.layers):
            self.dw[ln] = mc * self.dw[ln] + ((1 - mc) * lr) * grad[ln]['w'] 
            self.db[ln] = mc * self.db[ln] + ((1 - mc) * lr) * grad[ln]['b']
            layer.np['w'] -= self.dw[ln]
            layer.np['b'] -= self.db[ln]
        return None

class TrainGDA(TrainGD):
    """
    Gradient descent with adaptive learning rate
    
    :Support networks:
        newff (multi-layers perceptron)
    :Parameters:
        input: array like (l x net.ci)
            train input patterns
        target: array like (l x net.co)
            train target patterns
        epochs: int (default 500)
            Number of train epochs
        show: int (default 100)
            Print period
        goal: float (default 0.01)
            The goal of train
        lr: float (defaults 0.01)
            learning rate
        adapt: bool (detault False)
            type of learning
        lr_inc: float (> 1, default 1.05)
            Ratio to increase learning rate
        lr_dec: float (< 1, default 0.7)
            Ratio to decrease learning rate
        max_perf_inc:float (> 1, default 1.04)
            Maximum performance increase
    
    """
    def __init__(self, net, input, target, lr=0.01, adapt=False, lr_inc=1.05, 
                                                lr_dec=0.7, max_perf_inc=1.04):
        super(TrainGDA, self).__init__(net, input, target, lr, adapt)
        self.lr_inc = lr_inc
        self.lr_dec = lr_dec
        self.max_perf_inc = max_perf_inc
        self.err = []

    def learn(self, net, grad):
        #print 'GDA.learn'
        if len(self.err) > 1:
            f = self.err[-1] / self.err[-2]
            if f > self.max_perf_inc:
                self.lr *= self.lr_dec
            elif f < 1:
                self.lr *= self.lr_inc
        super(TrainGDA, self).learn(net, grad)
        return None
    
    def error(self, *args, **kwargs):
        e = super(TrainGDA, self).error(*args, **kwargs)
        self.err.append(e)
        return e

class TrainGDX(TrainGDA, TrainGDM):
    """
    Gradient descent with momentum backpropagation and adaptive lr
    
    :Support networks:
        newff (multi-layers perceptron)
    :Рarameters:
        input: array like (l x net.ci)
            train input patterns
        target: array like (l x net.co)
            train target patterns
        epochs: int (default 500)
            Number of train epochs
        show: int (default 100)
            Print period
        goal: float (default 0.01)
            The goal of train
        lr: float (defaults 0.01)
            learning rate
        adapt: bool (detault False)
            type of learning
        lr_inc: float (default 1.05)
            Ratio to increase learning rate
        lr_dec: float (default 0.7)
            Ratio to decrease learning rate
        max_perf_inc:float (default 1.04)
            Maximum performance increase
        mc: float (default 0.9)
            Momentum constant
    
    """
    def __init__(self, net, input, target, lr=0.01, adapt=False, lr_inc=1.05, 
                                        lr_dec=0.7, max_perf_inc=1.04, mc=0.9):
        """ init gdm"""
        super(TrainGDX, self).__init__(net, input, target, lr, adapt, lr_inc, 
                                        lr_dec, max_perf_inc)
        self.mc = mc
        
    
    
class TrainRprop(TrainGD2):
    """
    Resilient Backpropagation
    
    :Support networks:
        newff (multi-layers perceptron)
    :Parameters:
        input: array like (l x net.ci)
            train input patterns
        target: array like (l x net.co)
            train target patterns
        epochs: int (default 500)
            Number of train epochs
        show: int (default 100)
            Print period
        goal: float (default 0.01)
            The goal of train
        lr: float (defaults 0.07)
            learning rate (init rate)
        adapt bool (default False)
            type of learning
        rate_dec: float (default 0.5)
            Decrement to weight change
        rate_inc: float (default 1.2)
            Increment to weight change
        rate_min: float (default 1e-9)
            Minimum performance gradient
        rate_max: float (default 50)
            Maximum weight change
    
    """
    
    def __init__(self, net, input, target, lr=0.07, adapt=False, 
                    rate_dec=0.5, rate_inc=1.2, rate_min=1e-9, rate_max=50):
        
        super(TrainRprop, self).__init__(net, input, target, lr, adapt)
        self.rate_inc = rate_inc
        self.rate_dec = rate_dec
        self.rate_max = rate_max
        self.rate_min = rate_min
        size = tool.np_size(net)
        self.grad_prev = np.zeros(size)
        self.rate =  np.zeros(size) + lr
    
    def learn(self, net, grad):
    
        prod = grad * self.grad_prev
        # Sign not change
        ind = prod > 0 
        self.rate[ind] *= self.rate_inc
        # Sign change
        ind = prod < 0
        self.rate[ind] *= self.rate_dec
        
        self.rate[self.rate > self.rate_max] = self.rate_max
        self.rate[self.rate < self.rate_min] = self.rate_min
        
        self.x -= self.rate * np.sign(grad)
        self.grad_prev = grad
        return None

class TrainRpropM(TrainRprop):
    """
    Resilient Backpropagation Modified
    (with back-step when grad change sign)
    :Support networks:
        newff (multi-layers perceptron)
    :Parameters:
        input: array like (l x net.ci)
            train input patterns
        target: array like (l x net.co)
            train target patterns
        epochs: int (default 500)
            Number of train epochs
        show: int (default 100)
            Print period
        goal: float (default 0.01)
            The goal of train
        lr: float (defaults 0.07)
            learning rate (init rate)
        adapt bool (default False)
            type of learning
        rate_dec: float (default 0.5)
            Decrement to weight change
        rate_inc: float (default 1.2)
            Increment to weight change
        rate_min: float (default 1e-9)
            Minimum performance gradient
        rate_max: float (default 50)
            Maximum weight change
    
    """
    
    def learn(self, net, grad):
    
        prod = grad * self.grad_prev
        # Sign not change
        ind = prod > 0 
        self.rate[ind] *= self.rate_inc
        # Sign change
        ind = prod < 0
        # Back step
        self.x[ind] -= self.rate[ind] * np.sign(grad[ind])
        grad[ind] *= -1
        
        self.rate[ind] *= self.rate_dec
        
        self.rate[self.rate > self.rate_max] = self.rate_max
        self.rate[self.rate < self.rate_min] = self.rate_min
        
        self.x -= self.rate * np.sign(grad)
        self.grad_prev = grad
        return None

######gd>>>


def trainer(Train):
    """ Trainner init """
    from neurolab.core import Trainer
    #w = functools.wraps(Train)
    #c = w(Trainer(Train))
    c = Trainer(Train)
    c.__doc__ = Train.__doc__
    c.__name__ = Train.__name__
    c.__module__ = Train.__module__
    return c

# Initializing mains train functors
train_gd = trainer(TrainGD)
#train_gd2 = trainer(gd.TrainGD2)
train_gdm = trainer(TrainGDM)
train_gda = trainer(TrainGDA)
train_gdx = trainer(TrainGDX)
train_rprop = trainer(TrainRprop)
#train_rpropm = trainer(gd.TrainRpropM)

train_bfgs = trainer(TrainBFGS)
train_cg   = trainer(TrainCG)
train_ncg  = trainer(TrainNCG)

train_wta  = trainer(TrainWTA)
train_cwta = trainer(TrainCWTA)
train_lvq = trainer(TrainLVQ)
train_delta = trainer(TrainDelta)
