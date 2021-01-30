import torch

from .utils import from_config, flexible_wrapper, Incrementor
from .registry import register


@register("sampler")
class RandomSampler:
    """Random positive and negative example sampler.
    Adapted from
        https://github.com/pytorch/vision/blob/f16322b596c7dc9e9d67d3b40907694f29e16357/torchvision/models/detection/_utils.py#L10
    """
    @from_config(main_args="model->sampler", requires_all=True)
    def __init__(self, batch_size_per_image, positive_fraction):
        assert 0 <= positive_fraction <= 1
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    @flexible_wrapper()
    def __call__(self, anchor_labels):
        """Balancedly sample positive and negative samples from anchor labels.

        Parameters
        ----------
        anchor_labels : list[Tensor[long]]
            List of size `batch_size`, where i-th element is of shape (A_i,),
            containing the index of groundtruth boxes in the same image that
            each of the anchor box is mapped to.
        """
        pos_idx = []
        neg_idx = []
        for labels_per_image in anchor_labels:
            positive = torch.where(labels_per_image == 1)[0]
            negative = torch.where(labels_per_image == 0)[0]

            num_pos = min(
                round(self.batch_size_per_image * self.positive_fraction),
                positive.numel()
            )
            num_neg = min(
                self.batch_size_per_image - num_pos,
                round(num_pos * (1 - self.positive_fraction)
                      / self.positive_fraction),
                negative.numel(),
            )

            # Randomly select positive and negative examples
            perm1 = torch.randperm(
                positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(
                negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # Create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(
                labels_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                labels_per_image, dtype=torch.uint8
            )

            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx


@register("sampler")
class RandomSamplerWithHardNegativeMining(RandomSampler):
    """Random positive and negative example sampler with hard negative mining.

    Parameters
    ----------
    batch_size_per_image : int
        Total number of positive and negative examples to sample. Note that the
        final number might be less than this, since there may be not enough
        positive/negative examples.
    positive_fraction : float
        Fraction of positive examples out of total number of examples to
        sample. Note that the final fraction might be different from this.
    hard_fraction : float or dict or None
        If None, all negative examples will be "hard".
        If float, this will be the fraction of **hard** negative examples out
        of total number of negative examples.
        If dict, an `Incrementor` object will be initializer with this dict as
        keyword arguments.
    """
    @from_config(main_args="model->sampler", requires_all=True)
    def __init__(self, batch_size_per_image, positive_fraction,
                 hard_fraction=None):
        assert 0 <= positive_fraction <= 1
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

        if isinstance(hard_fraction, dict):
            self.hard_fraction = Incrementor(
                self.config, main_args="model->sampler->kwargs->hard_fraction")
        elif hard_fraction is None:
            self.hard_fraction = 1.0
        else:
            assert 0.0 < self.hard_fraction <= 1.0
            self.hard_fraction = hard_fraction

    def _get_hard_fraction(self):
        if isinstance(self.hard_fraction, Incrementor):
            curr_value = self.hard_fraction.get()
            self.hard_fraction.step()
        else:
            curr_value = self.hard_fraction
        return curr_value

    @flexible_wrapper()
    def __call__(self, anchor_labels, pred_objectness):
        """Balancedly sample positive and hard negative samples from anchor
        labels.

        Parameters
        ----------
        anchor_labels : list[Tensor[long]]
            List of size `batch_size`, where i-th element is of shape (A_i,),
            containing the index of groundtruth boxes in the same image that
            each of the anchor box is mapped to.
        pred_objectness : list[Tensor[float]]
            Same size and shape as `anchor_labels`. Objectness score of each
            of the corresponding anchor box.
        """
        hard_fraction = self._get_hard_fraction()
        assert 0 <= hard_fraction <= 1
        if hard_fraction == 0:
            return super(RandomSamplerWithHardNegativeMining, self).__call__(
                anchor_labels=anchor_labels)

        assert len(anchor_labels) == len(pred_objectness)
        pos_idx = []
        neg_idx = []

        for labels_per_image, pred_objectness_per_image in \
                zip(anchor_labels, pred_objectness):
            assert len(labels_per_image) == len(pred_objectness_per_image)

            # Randomly select positive examples
            positive = torch.where(labels_per_image == 1)[0]
            num_pos = min(
                round(self.batch_size_per_image * self.positive_fraction),
                positive.numel()
            )
            perm_pos = torch.randperm(
                positive.numel(), device=positive.device)[:num_pos]
            pos_idx_per_image = positive[perm_pos]

            # Hard-mine + randomly sample negative examples
            negative = torch.where(labels_per_image == 0)[0]
            num_neg = min(
                self.batch_size_per_image - num_pos,
                round(num_pos * (1 - self.positive_fraction)
                      / self.positive_fraction),
                negative.numel(),
            )
            num_hard_neg = round(num_neg * hard_fraction)
            num_nonhard_neg = num_neg - num_hard_neg
            _, sort_idxs = torch.sort(
                pred_objectness_per_image[negative], descending=True)
            negative = negative[sort_idxs]  # sorted by objectness scores
            # Hard-mine
            negative_hard = negative[:num_hard_neg]
            # Ramdonly sample the rest
            perm_neg = torch.randperm(
                len(negative) - num_hard_neg, device=negative.device
            ) + num_hard_neg  # index starts from `num_hard_neg`
            negative_non_hard = negative[perm_neg[:num_nonhard_neg]]
            # Combine
            neg_idx_per_image = torch.cat([negative_hard, negative_non_hard])

            # Create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(
                labels_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                labels_per_image, dtype=torch.uint8
            )

            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx
