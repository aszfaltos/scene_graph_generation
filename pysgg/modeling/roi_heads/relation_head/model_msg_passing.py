# modified from https://github.com/rowanz/neural-motifs
import copy

import torch
from torch import nn
from torch.nn import functional as F

from pysgg.config import cfg
from pysgg.data import get_dataset_statistics
from pysgg.modeling.make_layers import make_fc
from pysgg.modeling.roi_heads.relation_head.utils_relation import get_box_pair_info, get_box_info, \
    layer_init
from pysgg.modeling.utils import cat
from pysgg.structures.boxlist_ops import squeeze_tensor
from .utils_motifs import obj_edge_vectors, encode_box_info

# Import PCE infrastructure
from pysgg.modeling.roi_heads.relation_head.rel_proposal_network.models import (
    make_relation_confidence_aware_module,
)


class LearnableRelatednessGatingIMP(nn.Module):
    """Learnable linear scaling for relatedness scores (IMP-specific)."""
    def __init__(self):
        super(LearnableRelatednessGatingIMP, self).__init__()
        cfg_weight = cfg.MODEL.ROI_RELATION_HEAD.IMP_MODULE.LEARNABLE_SCALING_WEIGHT
        self.alpha = nn.Parameter(torch.Tensor([cfg_weight[0]]), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor([cfg_weight[1]]), requires_grad=False)

    def forward(self, relness):
        relness = torch.clamp(self.alpha * relness - self.alpha * self.beta, min=0, max=1.0)
        return relness


class SigmoidLearnableRelatednessGatingIMP(nn.Module):
    """Sigmoid-based learnable scaling for relatedness scores (IMP-specific)."""
    def __init__(self):
        super(SigmoidLearnableRelatednessGatingIMP, self).__init__()
        cfg_weight = cfg.MODEL.ROI_RELATION_HEAD.IMP_MODULE.LEARNABLE_SCALING_WEIGHT
        self.alpha = nn.Parameter(torch.Tensor([cfg_weight[0]]), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor([cfg_weight[1]]), requires_grad=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, relness):
        relness = self.alpha * (relness - self.beta)
        relness = self.sigmoid(relness)
        return torch.clamp(relness, min=0, max=1.0)


class PolinomialLearnableRelatednessGatingIMP(nn.Module):
    """Polynomial (smoothstep) learnable scaling for relatedness scores (IMP-specific)."""
    def __init__(self):
        super(PolinomialLearnableRelatednessGatingIMP, self).__init__()
        cfg_weight = cfg.MODEL.ROI_RELATION_HEAD.IMP_MODULE.LEARNABLE_SCALING_WEIGHT
        self.alpha = nn.Parameter(torch.Tensor([cfg_weight[0]]), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor([cfg_weight[1]]), requires_grad=True)

    def forward(self, relness):
        t = torch.clamp(self.alpha * (relness - self.beta), 0, 1)
        return t * t * (3 - 2 * t)


class IMPContext(nn.Module):
    def __init__(self, config, in_channels, hidden_dim=512, num_iter=3):
        super(IMPContext, self).__init__()
        self.cfg = config

        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.hidden_dim = hidden_dim
        self.num_iter = num_iter

        self.pairwise_feature_extractor = PairwiseFeatureExtractor(config,
                                                                   in_channels)

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        self.obj_unary = make_fc(self.pooling_dim, hidden_dim)
        self.edge_unary = make_fc(self.pooling_dim, hidden_dim)

        self.edge_gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.node_gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)

        self.sub_vert_w_fc = nn.Sequential(make_fc(hidden_dim * 2, 1), nn.Sigmoid())
        self.obj_vert_w_fc = nn.Sequential(make_fc(hidden_dim * 2, 1), nn.Sigmoid())
        self.out_edge_w_fc = nn.Sequential(make_fc(hidden_dim * 2, 1), nn.Sigmoid())
        self.in_edge_w_fc = nn.Sequential(make_fc(hidden_dim * 2, 1), nn.Sigmoid())

        # ============ PCE (Predicate Confidence Estimation) Configuration ============
        self.rel_aware_on = self.cfg.MODEL.ROI_RELATION_HEAD.IMP_MODULE.RELATION_CONFIDENCE_AWARE
        self.apply_gt_for_rel_conf = self.cfg.MODEL.ROI_RELATION_HEAD.IMP_MODULE.APPLY_GT
        self.filter_the_mp_instance = self.cfg.MODEL.ROI_RELATION_HEAD.IMP_MODULE.MP_ON_VALID_PAIRS
        self.relness_weighting_mp = self.cfg.MODEL.ROI_RELATION_HEAD.IMP_MODULE.RELNESS_MP_WEIGHTING
        self.vail_pair_num = self.cfg.MODEL.ROI_RELATION_HEAD.IMP_MODULE.MP_VALID_PAIRS_NUM

        self.mp_pair_refine_iter = 1
        self.relation_conf_aware_models = None
        self.pretrain_pre_clser_mode = False

        if self.rel_aware_on:
            # Number of PCE refinement iterations
            self.mp_pair_refine_iter = self.cfg.MODEL.ROI_RELATION_HEAD.IMP_MODULE.ITERATE_MP_PAIR_REFINE
            assert self.mp_pair_refine_iter > 0

            self.shared_pre_rel_classifier = (
                self.cfg.MODEL.ROI_RELATION_HEAD.IMP_MODULE.SHARE_RELATED_MODEL_ACROSS_REFINE_ITER
            )

            if self.mp_pair_refine_iter <= 1:
                self.shared_pre_rel_classifier = False

            # Build relation confidence aware models
            if not self.shared_pre_rel_classifier:
                self.relation_conf_aware_models = nn.ModuleList()
                for ii in range(self.mp_pair_refine_iter):
                    if ii == 0:
                        input_dim = self.pooling_dim
                    else:
                        input_dim = self.hidden_dim
                    self.relation_conf_aware_models.append(
                        make_relation_confidence_aware_module(input_dim)
                    )
            else:
                input_dim = self.pooling_dim
                self.relation_conf_aware_models = make_relation_confidence_aware_module(input_dim)

            # Relatedness score recalibration
            self.relness_score_recalibration_method = (
                self.cfg.MODEL.ROI_RELATION_HEAD.IMP_MODULE.RELNESS_MP_WEIGHTING_SCORE_RECALIBRATION_METHOD
            )

            if self.relness_score_recalibration_method == "learnable_scaling":
                self.learnable_relness_score_gating_recalibration = (
                    LearnableRelatednessGatingIMP()
                )
            elif self.relness_score_recalibration_method == "sigmoid_scaling":
                self.learnable_relness_score_gating_recalibration = (
                    SigmoidLearnableRelatednessGatingIMP()
                )
            elif self.relness_score_recalibration_method == "polynomial_scaling":
                self.learnable_relness_score_gating_recalibration = (
                    PolinomialLearnableRelatednessGatingIMP()
                )
            elif self.relness_score_recalibration_method == "minmax":
                self.min_relness = nn.Parameter(torch.Tensor([1e-5]), requires_grad=False)
                self.max_relness = nn.Parameter(torch.Tensor([0.5]), requires_grad=False)
            else:
                raise ValueError(
                    "Invalid relness_score_recalibration_method: "
                    + self.relness_score_recalibration_method
                )

    def set_pretrain_pre_clser_mode(self, val=True):
        """Enable/disable pre-classifier pretraining mode."""
        self.pretrain_pre_clser_mode = val

    def normalize(self, each_img_relness, selected_rel_prop_pairs_idx):
        """MinMax normalization with moving average for relatedness scores."""
        if len(squeeze_tensor(torch.nonzero(each_img_relness != 1.0))) > 10:
            select_relness_for_minmax = each_img_relness[selected_rel_prop_pairs_idx]
            curr_relness_max = select_relness_for_minmax.detach()[
                int(len(select_relness_for_minmax) * 0.05):
            ].max()
            curr_relness_min = select_relness_for_minmax.detach().min()

            min_val = self.min_relness.data * 0.7 + curr_relness_min * 0.3
            max_val = self.max_relness.data * 0.7 + curr_relness_max * 0.3

            if self.training:
                # Moving average for relness scores normalization
                self.min_relness.data = self.min_relness.data * 0.9 + curr_relness_min * 0.1
                self.max_relness.data = self.max_relness.data * 0.9 + curr_relness_max * 0.1
        else:
            min_val = self.min_relness
            max_val = self.max_relness

        def minmax_norm(data, min_v, max_v):
            return (data - min_v) / (max_v - min_v + 1e-5)

        # Apply on all non 1.0 relness scores
        each_img_relness[each_img_relness != 1.0] = torch.clamp(
            minmax_norm(each_img_relness[each_img_relness != 1.0], min_val, max_val),
            max=1.0,
            min=0.0,
        )
        return each_img_relness

    def ranking_minmax_recalibration(self, each_img_relness, selected_rel_prop_pairs_idx):
        """Recalibrate relatedness scores using ranking-based minmax normalization."""
        each_img_relness = self.normalize(each_img_relness, selected_rel_prop_pairs_idx)

        # Set top 10% pairs to relness 1.0 (must-keep relationships)
        total_rel_num = len(selected_rel_prop_pairs_idx)
        each_img_relness[selected_rel_prop_pairs_idx[:int(total_rel_num * 0.1)]] += (
            1.0 - each_img_relness[selected_rel_prop_pairs_idx[:int(total_rel_num * 0.1)]]
        )
        return each_img_relness

    def relness_score_recalibration(self, each_img_relness, selected_rel_prop_pairs_idx):
        """Apply the configured recalibration method to relatedness scores."""
        if self.relness_score_recalibration_method == "minmax":
            each_img_relness = self.ranking_minmax_recalibration(
                each_img_relness, selected_rel_prop_pairs_idx
            )
        elif self.relness_score_recalibration_method in (
            "learnable_scaling", "sigmoid_scaling", "polynomial_scaling"
        ):
            each_img_relness = self.learnable_relness_score_gating_recalibration(
                each_img_relness
            )
        return each_img_relness

    def _prepare_adjacency_matrix_with_pce(
        self, num_objs, rel_pair_idxs, relatedness, device
    ):
        """
        Prepare adjacency matrices with PCE-based filtering/weighting.

        Returns:
            sub2rel: subject-to-relation adjacency matrix (potentially weighted)
            obj2rel: object-to-relation adjacency matrix (potentially weighted)
            selected_relness: relatedness scores for selected pairs
            selected_rel_prop_pairs_idx: indices of selected relationship pairs
        """
        obj_count = sum(num_objs)
        rel_count = sum([len(pair_idx) for pair_idx in rel_pair_idxs])

        # Build batch-wise concatenated rel_pair_idxs
        rel_inds_batch_cat = []
        offset = 0
        for pair_idx, num_obj in zip(rel_pair_idxs, num_objs):
            rel_ind_i = copy.deepcopy(pair_idx)
            rel_ind_i += offset
            offset += num_obj
            rel_inds_batch_cat.append(rel_ind_i)
        rel_inds_batch_cat = torch.cat(rel_inds_batch_cat, 0)

        # Initialize adjacency matrices
        sub2rel = torch.zeros(obj_count, rel_count, device=device).float()
        obj2rel = torch.zeros(obj_count, rel_count, device=device).float()

        if relatedness is None or not self.filter_the_mp_instance:
            # No filtering: use all pairs
            obj_offset = 0
            rel_offset = 0
            for pair_idx, num_obj in zip(rel_pair_idxs, num_objs):
                num_rel = pair_idx.shape[0]
                sub_idx = pair_idx[:, 0].contiguous().long().view(-1) + obj_offset
                obj_idx = pair_idx[:, 1].contiguous().long().view(-1) + obj_offset
                rel_idx = torch.arange(num_rel, device=device).long().view(-1) + rel_offset

                sub2rel[sub_idx, rel_idx] = 1.0
                obj2rel[obj_idx, rel_idx] = 1.0

                obj_offset += num_obj
                rel_offset += num_rel

            selected_rel_prop_pairs_idx = torch.arange(rel_count, device=device)
            selected_relness = None
            return sub2rel, obj2rel, selected_relness, selected_rel_prop_pairs_idx

        # With PCE filtering/weighting
        rel_prop_pairs_relness_batch = []
        for idx, (num_obj, rel_ind_i) in enumerate(zip(num_objs, rel_pair_idxs)):
            related_matrix = relatedness[idx]
            rel_prop_pairs_relness = related_matrix[rel_ind_i[:, 0], rel_ind_i[:, 1]]
            rel_prop_pairs_relness_batch.append(rel_prop_pairs_relness)

        # Process each image's relatedness scores
        offset = 0
        rel_prop_pairs_relness_sorted_idx = []
        rel_prop_pairs_relness_batch_update = []

        for idx, each_img_relness in enumerate(rel_prop_pairs_relness_batch):
            selected_rel_prop_pairs_relness, selected_rel_prop_pairs_idx = torch.sort(
                each_img_relness, descending=True
            )

            if self.apply_gt_for_rel_conf:
                # Add non-GT rel pairs dynamically according to GT rel num
                gt_rel_idx = squeeze_tensor(
                    torch.nonzero(selected_rel_prop_pairs_relness == 1.0)
                )
                pred_rel_idx = squeeze_tensor(
                    torch.nonzero(selected_rel_prop_pairs_relness < 1.0)
                )
                pred_rel_num = int(len(gt_rel_idx) * 0.2)
                pred_rel_num = min(pred_rel_num, len(pred_rel_idx))
                pred_rel_num = max(pred_rel_num, 5)
                selected_rel_prop_pairs_idx = torch.cat((
                    selected_rel_prop_pairs_idx[gt_rel_idx],
                    selected_rel_prop_pairs_idx[pred_rel_idx[:pred_rel_num]],
                ))
            else:
                # Select top-k pairs
                selected_rel_prop_pairs_idx = selected_rel_prop_pairs_idx[:self.vail_pair_num]

                if self.relness_weighting_mp and not self.pretrain_pre_clser_mode:
                    each_img_relness = self.relness_score_recalibration(
                        each_img_relness, selected_rel_prop_pairs_idx
                    )
                    selected_rel_prop_pairs_idx = squeeze_tensor(
                        torch.nonzero(each_img_relness > 0.0001)
                    )

            rel_prop_pairs_relness_batch_update.append(each_img_relness)
            rel_prop_pairs_relness_sorted_idx.append(selected_rel_prop_pairs_idx + offset)
            offset += len(each_img_relness)

        selected_rel_prop_pairs_idx = torch.cat(rel_prop_pairs_relness_sorted_idx, 0)
        rel_prop_pairs_relness_batch_cat = torch.cat(rel_prop_pairs_relness_batch_update, 0)

        # Build adjacency matrices with selected pairs
        sub2rel[
            rel_inds_batch_cat[selected_rel_prop_pairs_idx, 0],
            selected_rel_prop_pairs_idx,
        ] = 1
        obj2rel[
            rel_inds_batch_cat[selected_rel_prop_pairs_idx, 1],
            selected_rel_prop_pairs_idx,
        ] = 1

        # Apply soft weighting if enabled
        if self.relness_weighting_mp and not self.pretrain_pre_clser_mode:
            weights = rel_prop_pairs_relness_batch_cat[selected_rel_prop_pairs_idx]
            sub2rel[
                rel_inds_batch_cat[selected_rel_prop_pairs_idx, 0],
                selected_rel_prop_pairs_idx,
            ] = weights
            obj2rel[
                rel_inds_batch_cat[selected_rel_prop_pairs_idx, 1],
                selected_rel_prop_pairs_idx,
            ] = weights

        return sub2rel, obj2rel, rel_prop_pairs_relness_batch_cat, selected_rel_prop_pairs_idx

    def forward(
        self, inst_features, proposals, union_features, rel_pair_idxs,
        rel_gt_binarys=None, logger=None
    ):
        """
        Forward pass with optional PCE (Predicate Confidence Estimation).

        Args:
            inst_features: Instance ROI features
            proposals: List of BoxList proposals
            union_features: Union box features for relationships
            rel_pair_idxs: List of relationship pair indices
            rel_gt_binarys: Ground truth binary relationship matrices (optional)
            logger: Logger for debugging (optional)

        Returns:
            obj_rep: Final object representations
            rel_rep: Final relationship representations
            pre_cls_logits_each_iter: Pre-classifier logits from each PCE iteration (or None)
            relatedness_each_iters: Relatedness matrices from each iteration (or None)
        """
        num_objs = [len(b) for b in proposals]

        augment_obj_feat, rel_feats = self.pairwise_feature_extractor(
            inst_features, union_features, proposals, rel_pair_idxs
        )

        # Initialize representations
        obj_rep = self.obj_unary(augment_obj_feat)
        rel_rep = F.relu(self.edge_unary(rel_feats))

        obj_count = obj_rep.shape[0]
        rel_count = rel_rep.shape[0]
        device = obj_rep.device

        # Build global index mappings (needed for message passing)
        sub_global_inds = []
        obj_global_inds = []
        obj_offset = 0
        for pair_idx, num_obj in zip(rel_pair_idxs, num_objs):
            sub_idx = pair_idx[:, 0].contiguous().long().view(-1) + obj_offset
            obj_idx = pair_idx[:, 1].contiguous().long().view(-1) + obj_offset
            sub_global_inds.append(sub_idx)
            obj_global_inds.append(obj_idx)
            obj_offset += num_obj
        sub_global_inds = torch.cat(sub_global_inds, dim=0)
        obj_global_inds = torch.cat(obj_global_inds, dim=0)

        # Track outputs for each refinement iteration
        relatedness_each_iters = []
        pre_cls_logits_each_iter = []
        refine_obj_feats = [obj_rep]
        refine_rel_feats = [rel_feats]  # Store original pooling_dim features for PCE

        # ============ Main PCE + Message Passing Loop ============
        for refine_iter in range(self.mp_pair_refine_iter):

            # --- Step 1: Compute PCE (if enabled) ---
            pre_cls_logits = None
            pred_relatedness_scores = None

            if self.rel_aware_on:
                input_features = refine_rel_feats[-1]

                if not self.shared_pre_rel_classifier:
                    pre_cls_logits, pred_relatedness_scores = self.relation_conf_aware_models[
                        refine_iter
                    ](input_features, proposals, rel_pair_idxs)
                else:
                    pre_cls_logits, pred_relatedness_scores = self.relation_conf_aware_models(
                        input_features, proposals, rel_pair_idxs
                    )
                pre_cls_logits_each_iter.append(pre_cls_logits)

            relatedness_scores = pred_relatedness_scores

            # Apply GT relatedness if configured
            if self.apply_gt_for_rel_conf and rel_gt_binarys is not None:
                ref_relatedness = [r.clone() for r in rel_gt_binarys]
                if pred_relatedness_scores is None:
                    relatedness_scores = ref_relatedness
                else:
                    relatedness_scores = pred_relatedness_scores
                    for idx, ref_rel in enumerate(ref_relatedness):
                        gt_rel_idx = ref_rel.nonzero()
                        if len(gt_rel_idx) > 0:
                            relatedness_scores[idx][gt_rel_idx[:, 0], gt_rel_idx[:, 1]] = 1.0

            relatedness_each_iters.append(relatedness_scores)

            # --- Step 2: Prepare adjacency matrices ---
            if self.rel_aware_on and self.filter_the_mp_instance and not self.pretrain_pre_clser_mode:
                # Use PCE-filtered adjacency matrices
                sub2rel, obj2rel, selected_relness, selected_pairs_idx = \
                    self._prepare_adjacency_matrix_with_pce(
                        num_objs, rel_pair_idxs, relatedness_scores, device
                    )
            else:
                # Standard adjacency matrices (no filtering)
                sub2rel = torch.zeros(obj_count, rel_count).to(device).float()
                obj2rel = torch.zeros(obj_count, rel_count).to(device).float()
                obj_offset = 0
                rel_offset = 0

                for pair_idx, num_obj in zip(rel_pair_idxs, num_objs):
                    num_rel = pair_idx.shape[0]
                    sub_idx = pair_idx[:, 0].contiguous().long().view(-1) + obj_offset
                    obj_idx = pair_idx[:, 1].contiguous().long().view(-1) + obj_offset
                    rel_idx = torch.arange(num_rel).to(device).long().view(-1) + rel_offset

                    sub2rel[sub_idx, rel_idx] = 1.0
                    obj2rel[obj_idx, rel_idx] = 1.0

                    obj_offset += num_obj
                    rel_offset += num_rel

            # --- Step 3: Iterative message passing ---
            # For iter=0, use obj_rep/rel_rep (already projected to hidden_dim by obj_unary/edge_unary)
            # For iter>0, refine_*_feats[-1] is already in hidden_dim (from GRU output)
            current_obj_rep = refine_obj_feats[-1] if refine_iter > 0 else obj_rep
            current_rel_rep = refine_rel_feats[-1] if refine_iter > 0 else rel_rep

            hx_obj = torch.zeros(obj_count, self.hidden_dim, requires_grad=False).to(device).float()
            hx_rel = torch.zeros(rel_count, self.hidden_dim, requires_grad=False).to(device).float()

            vert_factor = [self.node_gru(current_obj_rep, hx_obj)]
            edge_factor = [self.edge_gru(current_rel_rep, hx_rel)]

            for i in range(self.num_iter):
                # Compute edge context
                sub_vert = vert_factor[i][sub_global_inds]
                obj_vert = vert_factor[i][obj_global_inds]
                weighted_sub = self.sub_vert_w_fc(
                    torch.cat((sub_vert, edge_factor[i]), 1)) * sub_vert
                weighted_obj = self.obj_vert_w_fc(
                    torch.cat((obj_vert, edge_factor[i]), 1)) * obj_vert

                edge_factor.append(self.edge_gru(weighted_sub + weighted_obj, edge_factor[i]))

                # Compute vertex context
                pre_out = self.out_edge_w_fc(torch.cat((sub_vert, edge_factor[i]), 1)) * edge_factor[i]
                pre_in = self.in_edge_w_fc(torch.cat((obj_vert, edge_factor[i]), 1)) * edge_factor[i]
                vert_ctx = sub2rel @ pre_out + obj2rel @ pre_in
                vert_factor.append(self.node_gru(vert_ctx, vert_factor[i]))

            # Store refined features for next PCE iteration
            refine_obj_feats.append(vert_factor[-1])
            refine_rel_feats.append(edge_factor[-1])

        # --- Prepare outputs ---
        final_obj_rep = refine_obj_feats[-1]
        final_rel_rep = refine_rel_feats[-1]

        # Stack relatedness matrices for output (for visualization)
        if len(relatedness_each_iters) > 0 and relatedness_each_iters[0] is not None and not self.training:
            try:
                relatedness_each_iters = torch.stack(
                    [torch.stack(each) for each in relatedness_each_iters]
                )
                relatedness_each_iters = relatedness_each_iters.permute(1, 2, 3, 0)
            except (RuntimeError, TypeError, ValueError):
                # Stacking may fail if relatedness tensors have inconsistent shapes
                relatedness_each_iters = None
        else:
            relatedness_each_iters = None

        if len(pre_cls_logits_each_iter) == 0:
            pre_cls_logits_each_iter = None

        return final_obj_rep, final_rel_rep, pre_cls_logits_each_iter, relatedness_each_iters


class PairwiseFeatureExtractor(nn.Module):
    """
    extract the pairwise features from the object pairs and union features.
    most pipeline keep same with the motifs instead the lstm massage passing process
    """

    def __init__(self, config, in_channels):
        super(PairwiseFeatureExtractor, self).__init__()
        self.cfg = config
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        self.num_obj_classes = len(obj_classes)
        self.num_rel_classes = len(rel_classes)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        # features augmentation for instance features
        # word embedding
        # add language prior representation according to the prediction distribution
        # of objects
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.obj_dim = in_channels
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        self.word_embed_feats_on = self.cfg.MODEL.ROI_RELATION_HEAD.WORD_EMBEDDING_FEATURES
        if self.word_embed_feats_on:
            obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
            self.obj_embed_on_prob_dist = nn.Embedding(self.num_obj_classes, self.embed_dim)
            self.obj_embed_on_pred_label = nn.Embedding(self.num_obj_classes, self.embed_dim)
            with torch.no_grad():
                self.obj_embed_on_prob_dist.weight.copy_(obj_embed_vecs, non_blocking=True)
                self.obj_embed_on_pred_label.weight.copy_(obj_embed_vecs, non_blocking=True)
        else:
            self.embed_dim = 0

        # features augmentation for rel pairwise features
        self.rel_feature_type = config.MODEL.ROI_RELATION_HEAD.EDGE_FEATURES_REPRESENTATION

        # the input dimension is ROI head MLP, but the inner module is pooling dim, so we need
        # to decrease the dimension first.
        if self.pooling_dim != in_channels:
            self.rel_feat_dim_not_match = True
            self.rel_feature_up_dim = make_fc(in_channels, self.pooling_dim)
            layer_init(self.rel_feature_up_dim, xavier=True)
        else:
            self.rel_feat_dim_not_match = False

        self.pairwise_obj_feat_updim_fc = make_fc(self.hidden_dim + self.obj_dim + self.embed_dim,
                                                  self.hidden_dim * 2)

        self.outdim = self.pooling_dim
        # position embedding
        # encode the geometry information of bbox in relationships
        self.geometry_feat_dim = 128
        self.pos_embed = nn.Sequential(*[
            make_fc(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            make_fc(32, self.geometry_feat_dim), nn.ReLU(inplace=True),
        ])

        if self.rel_feature_type in ["obj_pair", "fusion"]:
            self.spatial_for_vision = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION
            if self.spatial_for_vision:
                self.spt_emb = nn.Sequential(*[make_fc(32, self.hidden_dim),
                                               nn.ReLU(inplace=True),
                                               make_fc(self.hidden_dim, self.hidden_dim * 2),
                                               nn.ReLU(inplace=True)
                                               ])
                layer_init(self.spt_emb[0], xavier=True)
                layer_init(self.spt_emb[2], xavier=True)

            self.pairwise_rel_feat_finalize_fc = nn.Sequential(
                make_fc(self.hidden_dim * 2, self.pooling_dim),
                nn.ReLU(inplace=True),
            )

        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.obj_hidden_linear = make_fc(self.obj_dim + self.embed_dim + self.geometry_feat_dim, self.hidden_dim)

        self.obj_feat_aug_finalize_fc = nn.Sequential(
            make_fc(self.hidden_dim + self.obj_dim + self.embed_dim, self.pooling_dim),
            nn.ReLU(inplace=True),
        )

        # untreated average features

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def pairwise_rel_features(self, augment_obj_feat, union_features, rel_pair_idxs, inst_proposals):
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in inst_proposals]
        num_objs = [len(p) for p in inst_proposals]
        # post decode
        # (num_objs, hidden_dim) -> (num_objs, hidden_dim * 2)
        # going to split single object representation to sub-object role of relationship
        pairwise_obj_feats_fused = self.pairwise_obj_feat_updim_fc(augment_obj_feat)
        pairwise_obj_feats_fused = pairwise_obj_feats_fused.view(pairwise_obj_feats_fused.size(0), 2, self.hidden_dim)
        head_rep = pairwise_obj_feats_fused[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = pairwise_obj_feats_fused[:, 1].contiguous().view(-1, self.hidden_dim)
        # split
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        # generate the pairwise object for relationship representation
        # (num_objs, hidden_dim) <rel pairing > (num_objs, hidden_dim)
        #   -> (num_rel, hidden_dim * 2)
        #   -> (num_rel, hidden_dim)
        obj_pair_feat4rel_rep = []
        pair_bboxs_info = []

        for pair_idx, head_rep, tail_rep, obj_box in zip(rel_pair_idxs, head_reps, tail_reps, obj_boxs):
            obj_pair_feat4rel_rep.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_bboxs_info.append(get_box_pair_info(obj_box[pair_idx[:, 0]], obj_box[pair_idx[:, 1]]))
        pair_bbox_geo_info = cat(pair_bboxs_info, dim=0)
        obj_pair_feat4rel_rep = cat(obj_pair_feat4rel_rep, dim=0)  # (num_rel, hidden_dim * 2)
        if self.spatial_for_vision:
            obj_pair_feat4rel_rep = obj_pair_feat4rel_rep * self.spt_emb(pair_bbox_geo_info)

        obj_pair_feat4rel_rep = self.pairwise_rel_feat_finalize_fc(obj_pair_feat4rel_rep)  # (num_rel, hidden_dim)

        return obj_pair_feat4rel_rep

    def forward(self, inst_roi_feats, union_features, inst_proposals, rel_pair_idxs, ):
        """

        :param inst_roi_feats: instance ROI features, list(Tensor)
        :param inst_proposals: instance proposals, list(BoxList())
        :param rel_pair_idxs:
        :return:
            obj_pred_logits obj_pred_labels 2nd time instance classification results
            obj_representation4rel, the objects features ready for the represent the relationship
        """
        # using label or logits do the label space embeddings
        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            obj_labels = cat([proposal.get_field("labels") for proposal in inst_proposals], dim=0)
        else:
            obj_labels = None

        if self.word_embed_feats_on:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                obj_embed_by_pred_dist = self.obj_embed_on_prob_dist(obj_labels.long())
            else:
                obj_logits = cat([proposal.get_field("predict_logits") for proposal in inst_proposals], dim=0).detach()
                obj_embed_by_pred_dist = F.softmax(obj_logits, dim=1) @ self.obj_embed_on_prob_dist.weight

        # box positive geometry embedding
        assert inst_proposals[0].mode == 'xyxy'
        pos_embed = self.pos_embed(encode_box_info(inst_proposals))

        # word embedding refine
        batch_size = inst_roi_feats.shape[0]
        if self.word_embed_feats_on:
            obj_pre_rep = cat((inst_roi_feats, obj_embed_by_pred_dist, pos_embed), -1)
        else:
            obj_pre_rep = cat((inst_roi_feats, pos_embed), -1)
        # object level contextual feature
        augment_obj_feat = self.obj_hidden_linear(obj_pre_rep)  # map to hidden_dim

        # todo reclassify on the fused object features
        # Decode in order
        if self.mode != 'predcls':
            # todo: currently no redo classification on embedding representation,
            #       we just use the first stage object prediction
            obj_pred_labels = cat([each_prop.get_field("pred_labels") for each_prop in inst_proposals], dim=0)
        else:
            assert obj_labels is not None
            obj_pred_labels = obj_labels

        # object labels space embedding from the prediction labels
        if self.word_embed_feats_on:
            obj_embed_by_pred_labels = self.obj_embed_on_pred_label(obj_pred_labels.long())

        # average action in test phrase for causal effect analysis
        if self.word_embed_feats_on:
            augment_obj_feat = cat((obj_embed_by_pred_labels, inst_roi_feats, augment_obj_feat), -1)
        else:
            augment_obj_feat = cat((inst_roi_feats, augment_obj_feat), -1)

        if self.rel_feature_type == "obj_pair" or self.rel_feature_type == "fusion":
            rel_features = self.pairwise_rel_features(augment_obj_feat, union_features,
                                                      rel_pair_idxs, inst_proposals)
            if self.rel_feature_type == "fusion":
                if self.rel_feat_dim_not_match:
                    union_features = self.rel_feature_up_dim(union_features)
                rel_features = union_features + rel_features

        elif self.rel_feature_type == "union":
            if self.rel_feat_dim_not_match:
                union_features = self.rel_feature_up_dim(union_features)
            rel_features = union_features

        else:
            assert False
        # mapping to hidden
        augment_obj_feat = self.obj_feat_aug_finalize_fc(augment_obj_feat)

        return augment_obj_feat, rel_features
