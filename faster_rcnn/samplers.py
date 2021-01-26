import torch


def get_sampler(sampler_name):
    samplers = {
        "random_sampler": RandomSampler,
    }
    if sampler_name not in samplers:
        raise ValueError(
            f"Invalid sampler name. Expected one of {list(samplers.keys())}, "
            f"got {sampler_name} instead.")
    return samplers[sampler_name]


class RandomSampler:
    """Random positive and negative example sampler.
    Adapted from
        https://github.com/pytorch/vision/blob/f16322b596c7dc9e9d67d3b40907694f29e16357/torchvision/models/detection/_utils.py#L10
    """
    def __init__(self, batch_size_per_image, positive_fraction):
        assert 0 <= positive_fraction <= 1
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, labels):
        pos_idx = []
        neg_idx = []
        for labels_per_image in labels:
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
