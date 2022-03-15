from .LearningRateFunctions import LearningRateFunc, \
                                   BasicLearningRateFunc
import numpy as np

# for realising velocities and other accumulated optimizer params
import gc


class Optimizer():
    # attributes:
    # function ptrs, abstract methods:
    # dumpCache = None
    # reinitialize = None
    # performParameterUpdate = None
    # onEpochEnd = None

    # forbid instances of this class
    def __new__(cls, *args, **kwargs):
        if cls is Optimizer:
            msg = "Objects of this class cannot be created.\n" \
                + "Use classes derived from this one.\n"
            raise TypeError(msg)
        return object.__new__(cls)

    def __init__(self, dumpCacheFunc, reinitializeFunc,
                 updateParametersFunc, onEpochEndFunc):
        # no args, just clears all but hyperparameters
        self.dumpCache = dumpCacheFunc
        # takes in layer param
        self.reinitialize = reinitializeFunc
        # takes in layers, gradients,
        # metrics_tracker [instance of class in ../MyAI/MetricsTracker.py]
        self.performParameterUpdate = updateParametersFunc
        # takes no args
        self.onEpochEnd = onEpochEndFunc

    # param decrease should include negative sign, like:
    # param_decrease: e.g. -alpha*dW
    # param: e.g. W; 
    def getParamUpdateRatio(param, param_decrease):
        # from: https://cs231n.github.io/neural-networks-3/
        param_scale = np.linalg.norm(param.ravel())
        update_scale = np.linalg.norm(param_decrease.ravel())
        # if ratio = ~1e-3 then good, else try increasing alpha
        ratio = update_scale / (1e-8 + param_scale)
        return ratio

    # param decrease should include negative sign, like:
    # param_decrease: e.g. -alpha*dW
    def storeParamUpdateRatio(param_key, layer,
                              layer_idx, param_decrease,
                              metrics_tracker):
        key = "{}_L{}".format(param_key, layer_idx)
        metrics_tracker["param_update_ratio"][key] \
            = Optimizer.getParamUpdateRatio(layer.params[param_key],
                                            param_decrease)


class GradientDescent(Optimizer):
    # attributes:
    # learning_rate_func = None

    def __init__(self, learning_rate):
        super().__init__(self.destroyCachedValues,
                         self.reinitialize,
                         self.updateParameters,
                         self.onEpochEnd)
        """ Learning rate can be a numeric value or instance
            of LearningRateFunc from LearningRateFunctions.py
        """
        self.setLearningRateFunc(learning_rate)
        # to be called from net when layers are constructed
        # self.reinitialize(layers)

    def setLearningRateFunc(self, learning_rate):
        """ Learning rate can be a numeric value or instance
            of LearningRateFunc from LearningRateFunctions.py
        """
        if not isinstance(learning_rate, LearningRateFunc):
            self.learning_rate_func = BasicLearningRateFunc(learning_rate)
        else:
            self.learning_rate_func = learning_rate

    def onEpochEnd(self):
        self.learning_rate_func.onEpochEnd()

    def reinitialize(self, layers=None):
        self.learning_rate_func.reinitialize()

    def destroyCachedValues(self):
        return

    # Magic:

    def updateParameters(self, layers, gradients,
                         metrics_tracker=None):
        """
            Updates parameters after single backward propagation.
            Takes in gradients dict for all layers, where key is
            of format: dW_Li/db_Li, where `i` is index of layer.
            Assumes layers is list of Layer objects from ../MyAI/Layers.py
        """
        for i in range(len(layers)):
            # do not update params dict in dropout layer since it does not
            # hold any weights
            if "W" in layers[i].params:
                param_change = -1 * self.learning_rate_func() * gradients["dW_L{}".format(i)]

                if not metrics_tracker is None \
                and "param_update_ratio" in metrics_tracker:
                    Optimizer.storeParamUpdateRatio("W", layers[i],
                                                    i, param_change,
                                                    metrics_tracker)

                layers[i].params["W"] += param_change

            if "b" in layers[i].params:
                param_change = -1 * self.learning_rate_func() * gradients["db_L{}".format(i)]

                if not metrics_tracker is None \
                and "param_update_ratio" in metrics_tracker:
                    Optimizer.storeParamUpdateRatio("b", layers[i],
                                                    i, param_change,
                                                    metrics_tracker)

                layers[i].params["b"] += param_change

        # mark the end of one update iteration of the net
        self.learning_rate_func.onIterationEnd()


class MomentumGradientDescent(Optimizer):
    # attributes:
    # learning_rate_func = None
    # velocities = {}
    # beta1 = None

    def __init__(self, learning_rate, beta1=0.9):
        """ Learning rate can be a numeric value or instance
            of LearningRateFunc from LearningRateFunctions.py
            If beta1 is 0 then this is just standard gradient descent.
        """
        super().__init__(self.destroyVelocities,
                         self.reinitialize,
                         self.updateParameters,
                         self.onEpochEnd)
        self.setLearningRateFunc(learning_rate)
        self.beta1 = beta1
        self.velocities = {}
        # to be called from net when layers are constructed:
        # self.reinitialize(layers)

    def setLearningRateFunc(self, learning_rate):
        """ Learning rate can be a numeric value or instance
            of LearningRateFunc from LearningRateFunctions.py
        """
        if not isinstance(learning_rate, LearningRateFunc):
            self.learning_rate_func = BasicLearningRateFunc(learning_rate)
        else:
            self.learning_rate_func = learning_rate

    def onEpochEnd(self):
        self.learning_rate_func.onEpochEnd()

    def destroyVelocities(self):
        del self.velocities
        self.velocities = None
        gc.collect()

    def reinitializeVelocities(self, layers):
        """
            Assumes layers is list of Layer objects from ../MyAI/Layer.py
            Builds a dictionary which contains the exponentially
            weighted average of the gradient.
        """
        self.velocities = {}
        for i in range(len(layers)):
            # do not update params dict in dropout layer since it does not
            # hold any weights
            if "W" in layers[i].params:
                self.velocities["dW_L{}".format(i)] \
                    = np.zeros_like(layers[i].params["W"])

            if "b" in layers[i].params:
                self.velocities["db_L{}".format(i)] \
                    = np.zeros_like(layers[i].params["b"])

    def reinitialize(self, layers):
        self.reinitializeVelocities(layers)
        self.learning_rate_func.reinitialize()

    # Magic:

    def updateParameters(self, layers, gradients,
                         metrics_tracker=None):
        """
            Updates parameters after single backward propagation.
            Also updates the velocity dict.
            Takes in gradients dict for all layers, where key is
            of format: dW_Li/db_Li, where `i` is index of layer.
            Assumes layers is list of Layer objects from ../MyAI/Layer.py
        """
        for i in range(len(layers)):
            self.updateLayerParameter("W", layers[i], i, gradients,
                                      metrics_tracker)
            self.updateLayerParameter("b", layers[i], i, gradients,
                                      metrics_tracker)

        # mark the end of one update iteration of the net
        self.learning_rate_func.onIterationEnd()

    def updateLayerParameter(self, param_key, layer, layer_idx,
                             gradients, metrics_tracker=None):
        """ param_key = "W" or "b" """
        # skip dropout layers since they have no params
        if not param_key in layer.params:
            return

        grad_key = "d{}_L{}".format(param_key, layer_idx)

        self.calcVelocity(grad_key, gradients)
        param_change = -1 * self.learning_rate_func() * self.velocities[grad_key]

        if not metrics_tracker is None \
        and "param_update_ratio" in metrics_tracker:
            Optimizer.storeParamUpdateRatio(param_key, layer,
                                       layer_idx, param_change,
                                       metrics_tracker)

        layer.params[param_key] += param_change

    def calcVelocity(self, gradient_key, gradients):
        """ gradient_key = dW_Li or db_Li """
        self.velocities[gradient_key] \
            =   self.beta1 * self.velocities[gradient_key] \
              + (1 - self.beta1) * gradients[gradient_key]


class Adam(Optimizer):
    # attributes:
    # learning_rate_func = None
    # velocities = {}
    # squared = {}
    # beta1 = None
    # beta2 = None
    # num_iter = None  # num of updates previously performed
    # epsilon = 1e-8

    epsilon = 1e-8  # to avoid division by 0

    def __init__(self, learning_rate, beta1=0.9, beta2=0.999):
        """ Learning rate can be a numeric value or instance
            of LearningRateFunc from LearningRateFunctions.py
            beta2 is recommended to be 0.999 by authors of Adam.
        """
        super().__init__(self.destroyWeightedAveragesCache, \
                         self.reinitialize,
                         self.updateParameters,
                         self.onEpochEnd)
        self.epsilon = Adam.epsilon
        self.setLearningRateFunc(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.num_iter = 1
        self.velocities = {}
        self.squared = {}
        # to be called from net when layers are constructed
        # self.reinitialize(layers)

    def setLearningRateFunc(self, learning_rate):
        """ Learning rate can be a numeric value or instance
            of LearningRateFunc from LearningRateFunctions.py
        """
        if not isinstance(learning_rate, LearningRateFunc):
            self.learning_rate_func = BasicLearningRateFunc(learning_rate)
        else:
            self.learning_rate_func = learning_rate

    def onEpochEnd(self):
        self.learning_rate_func.onEpochEnd()

    def destroyWeightedAveragesCache(self):
        """
            Frees self.velocities and self.squared.
        """
        del self.velocities
        del self.squared
        self.velocities = None
        self.squared = None
        self.num_iter = 1  # if 0 then there is division by 0
        gc.collect()

    def reinitialize(self, layers):
        """
            For the bias correction resets the count of the
            previously performed updates.
            Resets the weighted averages dictionaries.
        """
        self.reinitializeVelocities(layers)
        self.reinitializeSquared(layers)
        self.learning_rate_func.reinitialize()
        self.num_iter = 1

    def reinitializeVelocities(self, layers):
        """
            Assumes layers is list of Layer objects from ../MyAI/Layer.py
            Builds a dictionary which contains the exponentially
            weighted average of the gradient.
        """
        self.velocities = {}
        for i in range(len(layers)):
            # do not update params dict in dropout layer since it does not
            # hold any weights
            if "W" in layers[i].params:
                self.velocities["dW_L{}".format(i)] \
                    = np.zeros_like(layers[i].params["W"])

            if "b" in layers[i].params:
                self.velocities["db_L{}".format(i)] \
                    = np.zeros_like(layers[i].params["b"])

    def reinitializeSquared(self, layers):
        """
            Assumes layers is list of Layer objects from ../MyAI/Layer.py
            Builds a dictionary which contains the exponentially
            weighted average of the squared gradient.
        """
        self.squared = {}
        for i in range(len(layers)):
            # do not update params dict in dropout layer since it does not
            # hold any weights
            if "W" in layers[i].params:
                self.squared["dW_L{}".format(i)] \
                    = np.zeros_like(layers[i].params["W"])

            if "b" in layers[i].params:
                self.squared["db_L{}".format(i)] \
                    = np.zeros_like(layers[i].params["b"])

    def updateParameters(self, layers, gradients,
                         metrics_tracker=None):
        """
            Updates parameters after single backward propagation.
            Also updates velocities and squared dicts.
            For the bias correction keeps its own count of the
            previously performed updates which gets reset when
            reinitialized.
            Takes in gradients dict for all layers, where key is
            of format: dW_Li/db_Li, where `i` is index of layer.
            Assumes layers is list of Layer objects from ../MyAI/Layer.py
        """

        for i in range(len(layers)):
            self.updateLayerParameter("W", layers[i], i, gradients,
                                      metrics_tracker)
            self.updateLayerParameter("b", layers[i], i, gradients,
                                      metrics_tracker)

        # mark the end of one update iteration of the net
        self.num_iter += 1
        self.learning_rate_func.onIterationEnd()

    def updateLayerParameter(self, param_key, layer, layer_idx,
                             gradients, metrics_tracker=None):
        """ param_key = "W" or "b" """
        # skip dropout layers since they have no params
        if not param_key in layer.params:
            return

        grad_key = "d{}_L{}".format(param_key, layer_idx)

        # Moving average of the gradients:
        self.calcVelocity(grad_key, gradients)

        # Compute bias-corrected first moment estimate:
        v_corrected_dW = self.velocities[grad_key] \
                       / (1 + self.epsilon - self.beta1 ** self.num_iter)

        # Moving average of the gradients:
        self.squared[grad_key] = \
            self.beta2 * self.squared[grad_key] \
            + (1 - self.beta2) * (gradients[grad_key] ** 2)

        # Compute bias-corrected first moment estimate:
        s_corrected_dW = self.squared[grad_key] \
                       / (1 + self.epsilon - self.beta2 ** self.num_iter)

        param_change = -1 * self.learning_rate_func() * v_corrected_dW \
                     / (self.epsilon + np.sqrt(s_corrected_dW))

        if not metrics_tracker is None \
        and "param_update_ratio" in metrics_tracker:
            Optimizer.storeParamUpdateRatio(param_key, layer,
                                       layer_idx, param_change,
                                       metrics_tracker)

        layer.params[param_key] += param_change

    def calcVelocity(self, gradient_key, gradients):
        """ gradient_key = dW_Li or db_Li """
        self.velocities[gradient_key] \
            =   self.beta1 * self.velocities[gradient_key] \
              + (1 - self.beta1) * gradients[gradient_key]
