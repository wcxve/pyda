#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 04:31:52 2023

@author: xuewc
"""

import numpy as np
import pymc as pm
import pytensor as aesara
import pytensor.tensor as at

# Needed for integration
from scipy.integrate import quad, quad_vec
from pytensor.graph.op import Op
from pytensor.graph.basic import Apply
from pytensor import clone_replace

class Integrate(Op):

    # Class to perform integration of a scalar variable
    # on a bounded interval

    # Adapted from:
    # https://discourse.pymc.io/t/custom-theano-op-to-do-numerical-integration/734/12
    # With some modifications!

    def __init__(self, expr, var, *extra_vars):
        """
        Parameters
        ----------
        expr: Aesara Variable
            The expression encoding the output
        var: Aesara Variable
            The input variable
        """
        super().__init__()

        # function we're integrating
        self._expr = expr

        # input var we're integrating over
        self._var = var

        # other variables
        self._extra_vars = extra_vars

        # transform expression into callable function
        self._func = aesara.function(
            # a list with all the inputs
            [var] + list(extra_vars),
            # output
            self._expr,
            on_unused_input='ignore'
        )

    def make_node(self, start, stop, *extra_vars):

        self._extra_vars_node = extra_vars

        # make sure that the same number of extra variables
        # are passed here as were specified when defining the Op
        assert len(self._extra_vars) == len(extra_vars)

        # Define the bounds of integration
        self._start = start
        self._stop = stop

        # return an Apply instance with the input and output Variable
        return Apply(
            # op: The operation that produces `outputs` given `inputs`.
            op=self,
            # inputs: The arguments of the expression modeled by the `Apply` node.
            inputs=[start, stop] + list(extra_vars),
            # outputs: The outputs of the expression modeled by the `Apply` node.
            # NOTE: This is a scalar if self._expr is a scalar,
            # and a vector if self._expr is a vector. Etc.
            outputs=[self._expr.type()]
        )

    def perform(self, node, inputs, out):
        """
        Out is the output storage.
        Inputs are passed by value.
        A single output is returned indirectly
        as the first element of single-element lists (out)

        NOTE: There's a restriction, namely the variable to integrate
        has to be a scalar, even though the other variables can vector.

        Parameters
        ----------
        node: Apply node
            The output of make_node
        inputs: List of data
            The data can be operated on with numpy
        out: List
            output_storage is a list of storage cells where the output
            is to be stored. There is one storage cell for each output of the Op.
            The data put in output_storage must match the type of the symbolic output.
        """

        # Runs the computation in python
        start, stop, *args = inputs

        if self._expr.ndim == 0:
            val = quad(
                self._func,
                start,
                stop,
                args=tuple(args)
            )[0]
        elif self._expr.ndim == 1:
            # if the function is vector-valued
            # (e.g., the gradient of a vector),
            # use quad_vec
            val = quad_vec(
                self._func,
                start,
                stop,
                args=tuple(args)
            )[0]
        else:
            shape = self._func(
                start,
                *args
            ).shape

            def helpfunc(*args):
                return self._func(*args).flatten()

            val = quad_vec(
                helpfunc,
                start,
                stop,
                args=tuple(args)
            )[0].reshape(shape)

        # in-place modification of "out".
        # out is a single-element list
        out[0][0] = np.array(val)

    def grad(self, inputs, grads):
        """
        NOTE: This function does not calculate the gradient
        but rather implements part of the chain rule,
        i.e. multiplies the grads by the gradient wrt to the cost
        See https://aesara.readthedocs.io/en/latest/extending/op.html
        for an explanation

        Inputs in this case contains:
        [lower integration bound, upper integration bound, ...other variables of function]
        """

        # unpack the input
        start, stop, *args = inputs
        out, = grads

        # dictionary with the extra variables as keys
        # and the extra variables in "inputs" as values
        replace = dict(zip(
            self._extra_vars,
            args
        ))

        # The derivative of integral wrt to the upper limit of integration
        # is just the value of the function at that limit
        # (for lower limit, it's minus the function)
        # See e.g.,
        # https://math.stackexchange.com/questions/984111/differentiating-with-respect-to-the-limit-of-integration
        replace_ = replace.copy()
        replace_[self._var] = start
        dstart = out * clone_replace(
            # Clone a graph and replace subgraphs within it.
            # It returns a copy of the initial subgraph with the corresponding
            # substitutions.
            -self._expr,
            # Dictionary describing which subgraphs should be replaced by what.
            replace=replace_
        )

        replace_ = replace.copy()
        replace_[self._var] = stop
        dstop = out * clone_replace(
            self._expr,
            replace=replace_
        )

        # calculate the symbolic gradient of self._expr
        # wrt each extra variable.
        # This can be done because they're symbolic aesara variables!
        # This corresponds to the gradient of the expression
        # *inside* the integral (the inner part of Leibniz'
        # integral rule)
        grads = at.jacobian(
            # cost
            self._expr,
            # wrt
            self._extra_vars
        )

        dargs = []
        # loop over the gradients of the extra vars
        for grad in grads:

            # define an Apply node
            # for that gradient
            integrate = Integrate(
                grad,
                # variable we're integrating over
                self._var,
                *self._extra_vars
            )

            # Apply Leibniz' integral rule:
            # call integrate, which evaluates
            # the integral of the gradient.
            # And then multiply with previous gradient
            # that was passed in the input.
            # NOTE: This is not actually doing the operation,
            # but rather calling make_node, which *creates the node*
            # that does the operation
            darg = at.dot(
                integrate(
                    start, stop,
                    *args
                ).T,
                out
            )

            dargs.append(darg)

        # return a list with one Variable for each input in inputs
        return [dstart, dstop] + dargs
y_obs = np.array([8.3, 8.0, 7.8])
start = aesara.shared(1.)
stop = aesara.shared(2.)

with pm.Model() as basic_model:

    a = pm.Uniform('a', 1.5, 3.5)
    b = pm.Uniform(
        'b',
        4., 6.,
        shape=(3)
    )

    # Define the function to integrate in plain pytensor
    t = at.dscalar('t')
    t.tag.test_value = np.zeros(())

    a_ = at.dscalar('a_')
    a_.tag.test_value = np.ones(())*2.

    b_ = at.dvector('b_')
    b_.tag.test_value = np.ones((3))*5.

    func = t**a_ + b_
    integrate = Integrate(
        # Function we're integrating
        func,
        # variable we're integrating
        t,
        # other variables
        a_, b_
    )

    # Now we plug in the values from the model.
    # The `a_` and `b_` from above corresponds
    # to the `a` and `b` here.
    mu = integrate(
        start,
        stop,
        a,
        b
    )
    y = pm.Normal(
        'y',
        mu=mu,
        sigma=0.4,
        observed=y_obs
    )

with basic_model:
    trace = pm.sample(
        1500,
        tune=500,
        cores=2,
        chains=2,
        return_inferencedata=True
    )
