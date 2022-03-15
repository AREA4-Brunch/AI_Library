import logging


class MetricsTracker:
    # attributes:
    # metrics = {}
    # epoch_freq = 100
    # mini_batch_freq = 100
    # do_log_live = False
    # metrics_lifetime = "object's"

    valid_basic_metric_names = [
        "epoch_costs",
        "mini_batches_costs",
        "param_update_ratio"  # logged on epoch level
    ]
    valid_param_metric_names = [
        "accuracy"  # param is acc_func with args: y_correct, y_pred
    ]

    def __init__(self, basic_metrics_to_track=[],
                 metrics_to_track_with_params={},
                 epoch_freq=100, mini_batch_freq=100,
                 do_log_live=False, metrics_lifetime="object's"):
        """Sets metrics to track during training in NeuralNet.py

            Args:
                basic_metrics_to_track (list, optional):
                    Contains names of basic metrics to track.
                    Defaults to [].

                metrics_to_track_with_params (dict, optional):
                    Keys are names of metrics to track,
                    values are special args needed for those
                    metrics to be calculated.
                    Defaults to {}.

                do_log_live(bool, optional):
                    metrics will be logged as they are written

                metrics_lifetime(string, optional):
                    if equal to "log" nothing will be stored, only logged
                    if do_log_live is True.
                    otherwise the metrics will be stored within the object
        """
        self.metrics = {}
        self.do_log_live = do_log_live
        self.metrics_lifetime = metrics_lifetime

        self.setEpochFreq(epoch_freq)
        self.setMiniBatchFreq(mini_batch_freq)
        self.setBasicMetrics(basic_metrics_to_track)
        self.setParamMetrics(metrics_to_track_with_params)

    def __contains__(self, key):
        return key in self.metrics

    def __getitem__(self, key):
        return self.metrics[key]

    def clear(self):
        for key in MetricsTracker.valid_basic_metric_names:
            if key in self.metrics:
                self.metrics[key].clear()

        # parameterized need custom clearing
        if "accuracy" in self.metrics:
            self.metrics["accuracy"][1].clear()

    # Setters:

    def setBasicMetrics(self, metrics_to_track):
        MetricsTracker.validateMetricNames(metrics_to_track)
        for key in metrics_to_track:
            # into empty dict will the values regarding the metric
            # be inserted
            self.metrics[key] = {}

    def setParamMetrics(self, metrics_to_track):
        MetricsTracker.validateMetricNames(metrics_to_track)
        for key in metrics_to_track:
            # first arg are parameters stored under the key
            # into empty dict will the values regarding the metric
            # be inserted
            self.metrics[key] = ( metrics_to_track[key], {} )

    def setEpochFreq(self, epoch_freq):
        self.epoch_freq = epoch_freq

    def setMiniBatchFreq(self, mini_batch_freq):
        self.mini_batch_freq = mini_batch_freq

    # Static methods:

    def validateMetricNames(metrics_names):
        for key in metrics_names:
            if not key in MetricsTracker.valid_basic_metric_names \
            and not key in MetricsTracker.valid_param_metric_names: 
                msg = "key `{}` is not valid".format(key) \
                    + "\nfollowing keys are valid:\n{}\nwith params:\n{}".\
                        format(MetricsTracker.valid_basic_metric_names,
                               MetricsTracker.valid_param_metric_names)
                raise KeyError(msg)

    def updateMetricsMiniBatch(self, cost, losses, gradients,
                               mini_batch_idx, epoch_idx,
                               y_correct, X, neural_net_obj,
                               mini_batch_epoch_shuffle_seed):
        """ As args takes in all that is provided by the net
            during training, on each mini batch.
        """
        if mini_batch_idx % self.mini_batch_freq != 0:
            return

        mini_batch_key = "e({})_{}_m_{}".\
                format(mini_batch_epoch_shuffle_seed,
                       epoch_idx, mini_batch_idx)

        if "mini_batches_costs" in self.metrics:
            if self.do_log_live:
                logging.debug("{}: {}: {}".format(mini_batch_key,
                                                  "mini_batches_costs",
                                                  cost))

            if self.metrics_lifetime != "log":
                self.metrics["mini_batches_costs"][mini_batch_key] \
                    = cost

    def updateMetricsEpoch(self, epoch_cost, epoch_idx,
                           y_correct, X, neural_net_obj,
                           mini_batch_epoch_shuffle_seed):
        """ As args takes in all that is provided by the net
            during training, on each mini batch.
        """
        if epoch_idx % self.epoch_freq != 0:
            return

        epoch_key = "e({})_{}".\
            format(mini_batch_epoch_shuffle_seed, epoch_idx)

        if "epoch_costs" in self.metrics:
            if self.do_log_live:
                logging.debug("{}: {}: {}".format(epoch_key, "epoch_costs",
                                                  epoch_cost))

            if self.metrics_lifetime != "log":
                self.metrics["epoch_costs"][epoch_key] = epoch_cost

        if "accuracy" in self.metrics:
            # do not calculate accuracy if it won't be used
            if self.do_log_live or self.metrics_lifetime != "log":
                acc_func = self.metrics["accuracy"][0]
                accuracy = acc_func(y_correct,
                                    neural_net_obj.propagateForward(X, False))

                if self.do_log_live:
                    logging.debug("{}: {}: {}".format(epoch_key, "accuracy",
                                                      accuracy))
                if self.metrics_lifetime != "log":
                    self.metrics["accuracy"][1][epoch_key] = accuracy

        if "param_update_ratio" in self.metrics:
            # value was already set by optimizer, delete if lifetime
            # was just until it is logged
            if self.do_log_live:
                for key, ratio in self.metrics["param_update_ratio"].items():
                    logging.debug("{}: {}: {}".\
                        format("param_update_ratio", key, ratio))

            if self.metrics_lifetime == "log":
                self.metrics["param_update_ratio"].clear()
