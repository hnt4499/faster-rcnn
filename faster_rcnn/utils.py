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
