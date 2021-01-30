import time
import inspect

import torch


def smooth_l1_loss(input, target, beta=1. / 9, size_average=False):
    """Smmoth L1 loss, as defined in the Fast R-CNN paper.
        Girshick, R. (2015). Fast R-CNN.
    """
    diff = torch.abs(input - target)
    mask = (diff < beta)
    loss = torch.where(mask, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)

    if size_average:
        return loss.mean()
    return loss.sum()


def index_argsort(x, index, dim=-1):
    """Multi-indexer over multiple dimensions. The `index` tensor could be, for
    example, result of the `argsort` function. The first n shape of `x` tensor
    must be equal to the shape of `index`, i.e.,
        x.shape[index.ndim] == index.shape

    Parameters
    ----------
    dim : int
        Dimension over which argsort was performed.

    """
    x = x.transpose(0, dim)
    index = index.transpose(0, dim)
    if index.ndim < x.ndim:
        diff = x.ndim - index.ndim
        s = index.shape
        s = list(s) + [1] * diff
        index = index.view(*s)

    idxs = []
    for i, dim_ in enumerate(x.shape[1:][::-1]):
        idx = torch.arange(dim_)
        s = idx.shape
        s = list(s) + [1] * i
        idx = idx.view(*s)
        idxs.append(idx)

    idxs = [index] + idxs[::-1]
    x = x[tuple(idxs)].transpose(0, dim)
    return x


def apply_mask(x, mask):
    """Apply mask along the batch axis"""
    assert len(x) == len(mask)
    x_masked = []
    for x_i, mask_i in zip(x, mask):
        x_masked.append(x_i[~mask_i])
    return x_masked


def index_batch(x, idxs):
    """Index along the batch axis"""
    assert len(x) == len(idxs)
    x_indexed = []
    for x_i, idxs_i in zip(x, idxs):
        x_indexed.append(x_i[idxs_i])
    return x_indexed


def batching(function, inp):
    """Apply a function along the batch axis"""
    return [function(inp_i) for inp_i in inp]


def collect(config, args, collected):
    """Recursively collect each argument in `args` from `config` and write to
    `collected`."""
    if not isinstance(config, dict):
        return

    keys = list(config.keys())
    for arg in args:
        if arg in keys:
            if arg in collected:  # already collected
                raise RuntimeError(f"Found repeated argument: {arg}")
            collected[arg] = config[arg]

    for key, sub_config in config.items():
        collect(sub_config, args, collected)


def from_config(main_args=None, requires_all=False):
    """Wrapper for all classes, which wraps `__init__` function to take in only
    a `config` dict, and automatically collect all arguments from it. An error
    is raised when duplication is found.

    Parameters
    ----------
    main_args : str
        If specified (with "a->b" format), arguments will first be collected
        from this subconfig. If there are any arguments left, recursively find
        them in the entire config.
    requires_all : bool
        Whether all function arguments must be found in the config.
    """
    global_main_args = main_args
    if global_main_args is not None:
        global_main_args = global_main_args.split("->")

    def decorator(init):
        init_args = inspect.getfullargspec(init)[0][1:]  # excluding self

        def wrapper(self, config, main_args=None):
            # Add config to self
            self.config = config

            if main_args is None:
                main_args = global_main_args
            else:
                main_args = main_args.split("->")  # overwrite global_main_args

            collected = {}  # contains keyword arguments
            # Collect from main args
            if main_args is not None:
                sub_config = config
                for main_arg in main_args:
                    sub_config = sub_config[main_arg]
                collect(sub_config, init_args, collected)
            # Collect from the rest
            not_collected = [arg for arg in init_args if arg not in collected]
            collect(config, not_collected, collected)
            # Validate
            if requires_all and (len(collected) != len(init_args)):
                raise RuntimeError(
                    f"Found missing argument(s) when initializing "
                    f"{self.__class__.__name__} class. Expected {init_args}, "
                    f"collected {list(collected.keys())}.")
            # Call function
            return init(self, **collected)
        return wrapper
    return decorator


class IDict(dict):
    """Dict that returns `key` (i.e., identity) if `key` is not found."""
    def __getitem__(self, key):
        if key not in self:
            return key
        return super(IDict, self).__getitem__(key)


def flexible_wrapper(func_args=None, output_names=None, rename_args={}):
    """Wraps calls (`__call__` functions) so that it can take any keyword
    arguments. Note that the function is restricted to pass only keyword
    arguments to the `__call__` function, but NOT positinal arguments. Also,
    if `output_names` is not specified, the output of each `__call__` function
    must be again a dict to update the existing `kwargs` (see below).

    This wrapper is particularly useful for transformations (to ensure
    consistency between inputs and outputs). Nevertheless, it can be used for
    any `__call__` function.

    Parameters
    ----------
    func_args : list of str or None
        If specified, use this instead of inspecting the function being
        wrapped. This is useful for functions that take general inputs like
        `__call__(*args, **kwargs)`.
    output_names : list of str or None
        If not specified
            If the output of `func` is a dict, `rename_args` will be used to
            rename the output (if necessary).
            Otherwise, return the output as is.
        If specified, map each of `output_names` to each the output of `func`
        to form a new dict of output.
    rename_args : dict
        A dict containing arguments in source function and their respective
        name in target function. Used to rename args for consistency.
    """

    def decorator(func):
        if func_args is None:
            input_args = inspect.getfullargspec(
                func)[0][1:]  # not including `self`
        else:
            input_args = func_args

        to_new_args = IDict(rename_args)
        assert len(set(to_new_args.values())) == len(to_new_args)  # uniqueness

        def wrapper(self, *args, **kwargs):
            if len(args) > 0:
                raise TypeError(
                    "You must pass all arguments as keyword arguments.")
            # Get actual args
            actual_kwargs = {}
            for arg in input_args:
                if to_new_args[arg] in kwargs:
                    actual_kwargs[arg] = kwargs[to_new_args[arg]]
                    del kwargs[to_new_args[arg]]

            # Call function
            outp = func(self, **actual_kwargs)

            # Post process
            if output_names is not None:
                if len(output_names) == 1:
                    outp = {output_names[0]: outp}
                else:
                    assert len(output_names) == len(outp)
                    outp = dict(zip(output_names, outp))

            if isinstance(outp, dict):
                # Update existing kwargs
                kwargs.update(outp)

                # Convert to new args
                new_outp = {}
                for k, v in kwargs.items():
                    new_outp[to_new_args[k]] = v

                outp = new_outp

            return outp

        return wrapper

    return decorator


class Timer:
    def __init__(self):
        self.global_start_time = time.time()
        self.start_time = None
        self.last_interval = None
        self.accumulated_interval = None

    def start(self):
        assert self.start_time is None
        self.start_time = time.time()

    def end(self):
        assert self.start_time is not None
        self.last_interval = time.time() - self.start_time
        self.start_time = None

        # Update accumulated interval
        if self.accumulated_interval is None:
            self.accumulated_interval = self.last_interval
        else:
            self.accumulated_interval = (
                0.9 * self.accumulated_interval + 0.1 * self.last_interval)

    def get_last_interval(self):
        return self.last_interval

    def get_accumulated_interval(self):
        return self.accumulated_interval

    def get_total_time(self):
        return time.time() - self.global_start_time
