import torch.nn.functional as F
import torch

class SupervisedContrastiveLoss(torch.nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/abs/2004.11362.
    It computes the contrastive loss among labeled examples.
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        Args:
            features (concatenated list of tensors): A tensor with shape (K, C), where K = M1 + M2 + ... for minibatch of M1, M2, ... points with C features each.
            labels (concatenated list of tensors): A tensor with shape (K,), where K = M1 + M2 + ... corresponding to the pointcloud instance labels.
        """
        # Normalize feature vectors
        features = F.normalize(features, p=2, dim=1)
        
        # Compute mask for contrastive loss
        label_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        positive_mask = label_mask.fill_diagonal_(0).cuda(non_blocking=True)  # Remove self-contrast
        
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile labels to match anchor_contrast dimensions
        logits_mask = torch.scatter(
            torch.ones_like(label_mask).cuda(non_blocking=True),
            1,
            torch.arange(labels.size(0)).view(-1, 1).cuda(non_blocking=True),
            0
        ).cuda(non_blocking=True)
        
        exp_logits = torch.exp(logits) * logits_mask
        
        # Compute log_prob
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive
        log_prob = (positive_mask * log_prob).sum(1) / positive_mask.sum(1)

        # Loss
        loss = - (self.temperature / self.base_temperature) * log_prob
        loss = loss.mean()

        return loss
    


class TripletKLLoss(torch.nn.Module):
    def __init__(self, margin=1.0, reduction='mean'):
        super(TripletKLLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, anchor, positive, negative):
        """
        Compute the Triplet KL Divergence Loss.
        
        Parameters:
        - anchor: Tensor of shape (N, D), N is batch size, D is feature dimension
        - positive: Tensor of shape (N, D), N is batch size, D is feature dimension
        - negative: Tensor of shape (N, D), N is batch size, D is feature dimension
        
        Returns:
        - loss: Scalar Tensor, the computed loss.
        """

        # Ensure the input distributions are positive and sum to 1 (i.e., valid probability distributions)
        anchor_dist = F.softmax(anchor, dim=1)
        positive_dist = F.softmax(positive, dim=1)
        negative_dist = F.softmax(negative, dim=1)
        
        # Calculate KL Divergence between anchor and positive
        kl_pos = F.kl_div(anchor_dist.log(), positive_dist, reduction='none').sum(1)
        
        # Calculate KL Divergence between anchor and negative
        kl_neg = F.kl_div(anchor_dist.log(), negative_dist, reduction='none').sum(1)
        
        # Compute the triplet loss with margin
        losses = F.relu(kl_pos - kl_neg + self.margin)

        # Apply reduction
        if self.reduction == 'mean':
            loss = losses.mean()
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            return losses

        return loss


def average_cosine_distance(features, labels):
    unique_labels = torch.unique(labels)
    K = len(unique_labels)

    # Create a mask for each group
    masks = labels.unsqueeze(0) == unique_labels.unsqueeze(1)
    group_features = torch.matmul(masks.float(), features) / masks.sum(1, keepdim=True).float()
    # mean_group_feature = group_features / masks.sum(1, keepdim=True).float()

    # pos_cos_sim = F.cosine_similarity(
    #     group_features, mean_group_feature.repeat(group_features.size(0), 1))

    # Compute cosine similarity using broadcasting
    normalized_group_features = F.normalize(group_features, p=2, dim=1)
    cosine_similarity = torch.mm(normalized_group_features, normalized_group_features.t())

    # Convert to cosine distance and fill the diagonal with zeros
    distance_matrix = 1 - cosine_similarity

    return distance_matrix


def batch_aux_hinge_loss(feature_list, label_list, margin=0.05):
    batch_margin_loss = 0.0
    batch_pos_loss = 0.0
    for features, labels in zip(feature_list, label_list):
        # Normalize features for cos sim computation
        features = F.normalize(features, p=2, dim=-1)

        unique_labels = torch.unique(labels)
        K = len(unique_labels)

        masks = labels.unsqueeze(0) == unique_labels.unsqueeze(1)
        mean_features = torch.matmul(masks.float(), features) / masks.sum(1, keepdim=True).float()
        other_masks = ~masks
        K_mask = ~F.one_hot(torch.arange(0,K)).to(labels.device).bool()

        scene_margin_loss = 0.0
        scene_pos_loss = 0.0

        for k in range(K):
            # Positive samples
            mask_features = features[masks[k]]
            cos_sim = torch.mm(mask_features, mask_features.t())
            pos_cos_sim = cos_sim.mean()

            # Convert to cosine distance
            scene_pos_loss += 1.0 - pos_cos_sim
            
            # Negative samples - take mean from other labels
            other_features = K_mask[k].unsqueeze(1) * mean_features
            mask_features_tile = mask_features.unsqueeze(1)
            other_features_tile = other_features.unsqueeze(0)
            neg_cos_sim = F.cosine_similarity(
             mask_features_tile, other_features_tile, dim=2).mean()
            
            # Apply margin to the loss
            scene_margin_loss += torch.clip(
             -pos_cos_sim + neg_cos_sim + margin, 0)

        batch_margin_loss += scene_margin_loss / K
        batch_pos_loss += scene_pos_loss / K

    batch_margin_loss /= len(feature_list)  # Normalize by number of scenes in the batch
    batch_pos_loss /= len(feature_list)

    return batch_pos_loss, batch_margin_loss


def batch_auxiliary_loss(feature_list, label_list, margin=0.1):
    """
    Auxiliary loss that encourages diversity in the feature space for different objects
    across a batch of scenes, including both positive and negative similarity, with a margin.
    feature_list: List of Tensors, each of shape (M_i, C) representing the output features
                  from the encoder for each scene in the batch.
    label_list: List of Tensors, each of shape (M_i,) representing the unique object IDs
                for each point in each scene in the batch.
    margin: A float representing the margin threshold for the difference between positive
            and negative similarities.
    """
    batch_loss = 0.0
    for features, labels in zip(feature_list, label_list):
        unique_labels = torch.unique(labels)
        scene_loss = 0.0
        for label in unique_labels:
            mask = (labels == label)
            other_mask = (labels != label)
            if torch.sum(mask) > 1 and torch.sum(other_mask) > 0:  # Ensure valid positive and negative samples
                # Positive samples
                label_features = features[mask]
                mean_feature = torch.mean(label_features, dim=0, keepdim=True)
                positive_cos_sim = F.cosine_similarity(label_features, mean_feature.repeat(label_features.size(0), 1))

                # Negative samples
                other_features = features[other_mask]
                # Expand dims to compute pairwise cosine similarity
                expanded_label_features = label_features.unsqueeze(1)
                expanded_other_features = other_features.unsqueeze(0)
                print(expanded_label_features.shape, expanded_other_features.shape)
                negative_cos_sim = F.cosine_similarity(expanded_label_features, expanded_other_features, dim=2)

                # Apply margin to the loss
                scene_loss += torch.clip(
                    positive_cos_sim.mean() - negative_cos_sim.mean() - margin, 0)

        batch_loss += scene_loss
    
    return batch_loss / len(feature_list)  # Normalize by number of scenes in the batch


def batch_auxiliary_contrastive_loss(feature_list, label_list):
    total_loss = 0.0
    for features, labels in zip(feature_list, label_list):
        features_norm = F.normalize(features, p=2, dim=1)
        cosine_sim = torch.mm(features_norm, features_norm.t())

        labels = labels.unsqueeze(1)
        positive_mask = torch.eq(labels, labels.T).fill_diagonal_(0)
        negative_mask = ~positive_mask

        positive_loss = (1 - cosine_sim) * positive_mask.float()
        negative_loss = F.relu(cosine_sim) * negative_mask.float()  # Use ReLU to ensure non-negative values

        total_loss += positive_loss.mean() + negative_loss.mean()

    loss = total_loss / len(feature_list) 

    return loss



def batch_auxiliary_contrastive_loss(feature_list, label_list):
    """
    Calculate an auxiliary contrastive loss for a batch of scenes, enforcing similarity within same objects and diversity between different objects.
    
    :param feature_list: List of Tensors, each of shape (M_i, C) representing the encoder output features for each scene.
    :param label_list: List of Tensors, each of shape (M_i,) representing the object IDs for each point in each scene.
    :return: The auxiliary loss value for the batch.
    """
    total_positive_loss = 0.0
    total_negative_loss = 0.0
    total_positive_pairs = 0
    total_negative_pairs = 0
    
    for features, labels in zip(feature_list, label_list):
        # Normalize the feature vectors to have unit length
        features_norm = F.normalize(features, p=2, dim=1)
        
        # Calculate cosine similarity matrix
        cosine_sim = torch.mm(features_norm, features_norm.t())
        
        # Create a mask for positive samples (points within the same object)
        positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).to(labels.device)
        positive_mask = positive_mask.fill_diagonal_(0)  # Remove self-contrast
        
        # Create a mask for negative samples (points from different objects)
        negative_mask = ~positive_mask
        
        # Calculate loss for positive pairs (should be close to 1)
        positive_loss = (1 - cosine_sim) * positive_mask.float()
        
        # Calculate loss for negative pairs (should be close to -1 or 0)
        negative_loss = (1 - cosine_sim) * negative_mask.float()
        
        # Sum the losses for the current scene
        total_positive_loss += positive_loss.sum()
        total_negative_loss += negative_loss.sum()
        total_positive_pairs += positive_mask.float().sum()
        total_negative_pairs += negative_mask.float().sum()
    
    # Normalize the total losses by the number of pairs
    loss = (total_positive_loss / total_positive_pairs) + (total_negative_loss / total_negative_pairs)
    
    return loss


def cosine_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example. (B, C)
        targets (Tensor): A float tensor with the same shape as inputs (B, C)
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

    # p = torch.sigmoid(inputs)
    # ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    dloss = (1 - torch.nn.CosineSimilarity()
                        (inputs, targets))

    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss