from copy import deepcopy
from time import time
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from continual_learning.IncrementalApproach import IncrementalApproach
from continual_learning.metrics.metric_evaluator import MetricEvaluator
from continual_learning.models.BaseModel import BaseModel
from continual_learning.utils.buffer import Buffer
from continual_learning.utils.empirical_feature_matrix import EmpiricalFeatureMatrix
from continual_learning.utils.imbalance_loss import CBLoss, FocalLoss
from continual_learning.utils.proto import ProtoGenerator
from continual_learning.utils.training_utils import (
    compute_rotations,
    get_old_new_features,
    save_efm,
)


class CEOS(IncrementalApproach):
    def __init__(self, args, device, out_path, class_per_task, task_dict):
        super().__init__(args, device, out_path, class_per_task, task_dict)
        self.args = args
        self.model = BaseModel(backbone=self.backbone, dataset=args.dataset)
        self.old_model = None

        self.feature_matrix_lambda = args.ceos_lamb
        self.damping = args.ceos_damping
        self.prototype_update_sigma = args.ceos_protoupdate
        self.prototype_batch_size = args.ceos_protobatchsize

        self.proto_generator = ProtoGenerator(
            device=args.device,
            task_dict=task_dict,
            batch_size=args.batch_size,
            out_path=out_path,
            feature_space_size=self.model.get_feat_size(),
        )

        self.previous_efm = None
        self.print_running_approach()

        if self.args.cutmix:
            self.cutmix_alpha = getattr(args, "cutmix_alpha", 1.0)
            self.cutmix_lambda = getattr(args, "cutmix_lambda", 1.0)

    def print_running_approach(self):
        super().print_running_approach()
        print("\n ceos_hyperparams")
        print(f"- ceos_lamb: {self.feature_matrix_lambda}")
        print(f"- damping: {self.damping}")
        print("\n Proto_hyperparams")
        print(f"- sigma update prototypes {self.prototype_update_sigma}")

    def pre_train(self, task_id):
        if task_id == 0 and self.rotation:
            self.auxiliary_classifier = nn.Linear(512, len(self.task_dict[task_id]) * 3)
            self.auxiliary_classifier.to(self.device)
        else:
            self.auxiliary_classifier = None

        self.old_model = deepcopy(self.model)
        self.old_model.freeze_all()
        self.old_model.to(self.device)
        self.old_model.eval()

        self.model.add_classification_head(len(self.task_dict[task_id]))
        self.model.to(self.device)

        if task_id > 0:
            print("Using PR-ACE")
            self.buffer = Buffer(task_id, self.task_dict)
        else:
            print("Standard training with cross entropy")

        super().pre_train(task_id)

    def consolidation_loss(self, features, old_features):
        features = features.unsqueeze(1)
        old_features = old_features.unsqueeze(1)
        regularized_matrix = (
            self.feature_matrix_lambda * self.previous_efm
            + self.damping * torch.eye(self.previous_efm.shape[0], device=self.device)
        )
        drift = features - old_features
        return torch.mean(
            torch.bmm(
                torch.bmm(drift, regularized_matrix.expand(features.shape[0], -1, -1)),
                drift.permute(0, 2, 1),
            )
        )

    def _make_samples_per_class(self, task_id):
        if self.args.balanced_weight and not self.args.ascending_weight:
            return make_balanced_samples_per_class(
                class_per_task=self.class_per_task,
                task_id=task_id,
                current_samples=500,
                min_old_samples=self.args.start_old,
            )
        if self.args.ascending_weight:
            return make_incrementing_samples_per_class(
                class_per_task=self.class_per_task,
                task_id=task_id,
                total_tasks=self.args.n_task,
                current_samples=500,
                min_old_samples=self.args.start_old,
            )
        return make_samples_per_class(
            class_per_task=self.class_per_task,
            task_id=task_id,
            current_samples=500,
            old_samples=self.args.start_old,
        )

    def _prototype_and_classification_losses(
        self,
        outputs,
        targets,
        task_id,
        overall_logits,
        overall_targets,
        current_batch_size,
    ):
        current_targets = self.rescale_targets(targets[:current_batch_size], task_id)
        current_logits = outputs[task_id][:current_batch_size]

        if self.args.cb_loss:
            samples_per_cls = self._make_samples_per_class(task_id)
            loss_type = "focal" if self.args.focal_loss else "ce"
            loss_fn = CBLoss(
                samples_per_class=samples_per_cls,
                beta=0.9999,
                loss_type=loss_type,
                gamma=2.0,
            )
            prototype_loss = loss_fn(overall_logits, overall_targets)

            if self.args.focal_loss and not self.args.partial:
                start = task_id * self.class_per_task
                end = start + self.class_per_task
                current_loss_fn = CBLoss(
                    samples_per_cls[start:end],
                    beta=0.9999,
                    loss_type="focal",
                    gamma=2.0,
                )
                classification_loss = current_loss_fn(current_logits, current_targets)
            else:
                classification_loss = F.cross_entropy(current_logits, current_targets)

            return classification_loss, prototype_loss

        prototype_loss = F.cross_entropy(overall_logits, overall_targets)
        classification_loss = F.cross_entropy(current_logits, current_targets)
        return classification_loss, prototype_loss

    def train_criterion(
        self,
        outputs,
        targets,
        task_id,
        features,
        old_features,
        proto_to_samples,
        current_batch_size,
    ):
        classification_loss, prototype_loss, reg_loss, n_proto = 0, 0, 0, 0
        cutmix_loss = 0

        if task_id > 0:
            reg_loss = self.consolidation_loss(features[:current_batch_size], old_features)

            with torch.no_grad():
                proto_aug, proto_aug_label, _ = self.proto_generator.perturbe(
                    task_id, self.prototype_batch_size
                )
                proto_aug = proto_aug[:proto_to_samples].to(self.device)
                proto_aug_label = proto_aug_label[:proto_to_samples].to(self.device)
                n_proto = proto_to_samples

            if self.args.cutmix:
                proto_aug, proto_aug_label, feat_rnd, lab_rnd = random_match_proto(
                    proto_aug,
                    proto_aug_label,
                    features.detach(),
                    targets,
                )
                mixed_feat, lab_a, lab_b, lam_vec = self.feature_cutmix_batch(
                    proto_aug,
                    proto_aug_label,
                    feat_rnd,
                    lab_rnd,
                    alpha=self.cutmix_alpha,
                    contiguous=self.args.cutmix_contiguous,
                )
                mix_logits = torch.cat([head(mixed_feat) for head in self.model.heads], dim=1)
                ce_a = F.cross_entropy(mix_logits, lab_a, reduction="none")
                ce_b = F.cross_entropy(mix_logits, lab_b, reduction="none")
                cutmix_loss = torch.mean(lam_vec * ce_a + (1.0 - lam_vec) * ce_b)

            if self.args.eos:
                mix_proto_aug, mix_proto_aug_label = self.expansive_oversampling(
                    proto_aug,
                    proto_aug_label,
                    features[:current_batch_size],
                    targets[:current_batch_size],
                    k=self.args.eos_k,
                )
                proto_aug = torch.cat([proto_aug, mix_proto_aug], dim=0)
                proto_aug_label = torch.cat([proto_aug_label, mix_proto_aug_label], dim=0)
                n_proto += mix_proto_aug.shape[0]

            prototype_logits = torch.cat([head(proto_aug) for head in self.model.heads], dim=1)
            replay_logits = torch.cat(list(outputs.values()), dim=1)[current_batch_size:]
            overall_logits = torch.cat([prototype_logits, replay_logits], dim=0)
            overall_targets = torch.cat([proto_aug_label, targets[current_batch_size:]])

            classification_loss, prototype_loss = self._prototype_and_classification_losses(
                outputs,
                targets,
                task_id,
                overall_logits,
                overall_targets,
                current_batch_size,
            )
        else:
            if self.args.focal_loss and not self.args.partial:
                focal_loss_fn = FocalLoss(gamma=2.0, alpha=None, reduction="mean")
                classification_loss = focal_loss_fn(
                    torch.cat(list(outputs.values()), dim=1),
                    targets,
                )
            else:
                classification_loss = F.cross_entropy(torch.cat(list(outputs.values()), dim=1), targets)

        return classification_loss, reg_loss, prototype_loss, n_proto, cutmix_loss

    def train(self, task_id, trn_loader, epoch, epochs):
        if task_id == 0:
            self.model.train()
        else:
            self.model.eval()

        if task_id == 0 and self.rotation:
            self.auxiliary_classifier.train()

        start = time()
        count_batches = 0
        for images, targets in trn_loader:
            count_batches += 1

            images = images.to(self.device)
            targets = targets.to(self.device)
            current_batch_size = images.shape[0]

            if task_id == 0 and self.rotation:
                images_rot, target_rot = compute_rotations(
                    images,
                    self.image_size,
                    self.task_dict,
                    targets,
                    task_id,
                )
                images = torch.cat([images, images_rot], dim=0)
                targets = torch.cat([targets, target_rot], dim=0)

            if task_id > 0:
                _, old_features = self.old_model(images)
                if self.buffer.previous_batch_samples is not None:
                    proto_to_samples, previous_batch_samples, previous_batch_labels = self.buffer.sample(
                        self.batch_size
                    )
                    if not self.args.noreplay:
                        images = torch.cat([images, previous_batch_samples], dim=0)
                        targets = torch.cat([targets, previous_batch_labels], dim=0)
                else:
                    proto_to_samples = self.prototype_batch_size
            else:
                old_features = None
                proto_to_samples = 0

            outputs, features = self.model(images)

            if task_id == 0 and self.rotation:
                out_rot = self.auxiliary_classifier(features)
                outputs[task_id] = torch.cat([outputs[task_id], out_rot], axis=1)

            classification_loss, ceos_loss, prototype_loss, _, cutmix_loss = self.train_criterion(
                outputs,
                targets,
                task_id,
                features,
                old_features,
                current_batch_size=current_batch_size,
                proto_to_samples=proto_to_samples,
            )

            loss = (
                self.args.cls_lambda * classification_loss
                + self.args.ceos_lambda * ceos_loss
                + self.args.proto_lambda * prototype_loss
            )
            if self.args.cutmix:
                loss += self.cutmix_lambda * cutmix_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if task_id > 0:
                self.buffer.add_data(
                    current_samples=images[:current_batch_size],
                    current_targets=targets[:current_batch_size],
                )

        end = time()
        print(f"Task {task_id}, Epoch {epoch}/{epochs}, N_batch {count_batches}, Elapsed Time {end-start}s")

    def eval_criterion(self, outputs, targets, task_id, features, old_features):
        classification_loss, prototype_loss, reg_loss = 0, 0, 0
        n_proto = 0

        if task_id > 0:
            reg_loss = self.consolidation_loss(features, old_features)
            with torch.no_grad():
                proto_aug, proto_aug_label, n_proto = self.proto_generator.perturbe(
                    task_id,
                    self.prototype_batch_size,
                )
                proto_aug = proto_aug.to(self.device)
                proto_aug_label = proto_aug_label.to(self.device)
                prototype_logits = torch.cat([head(proto_aug) for head in self.model.heads], dim=1)
                prototype_loss = F.cross_entropy(prototype_logits, proto_aug_label)

        classification_loss = F.cross_entropy(torch.cat(list(outputs.values()), dim=1), targets)
        return classification_loss, reg_loss, prototype_loss, n_proto

    def eval(self, current_training_task, test_id, loader, epoch, verbose):
        metric_evaluator = MetricEvaluator(self.out_path, self.task_dict)
        classification_loss, ceos_loss, prototype_loss = 0, 0, 0
        n_samples, total_prototypes = 0, 0

        with torch.no_grad():
            self.old_model.eval()
            self.model.eval()

            for images, targets in loader:
                images = images.to(self.device)
                targets = targets.type(dtype=torch.int64).to(self.device)

                current_batch_size = images.shape[0]
                original_labels = deepcopy(targets)

                outputs, features = self.model(images)
                _, old_features = self.old_model(images)
                cls_loss_batch, ceos_loss_batch, proto_loss_batch, n_proto = self.eval_criterion(
                    outputs,
                    targets,
                    current_training_task,
                    features,
                    old_features,
                )

                classification_loss += cls_loss_batch * current_batch_size
                ceos_loss += ceos_loss_batch * current_batch_size
                prototype_loss += proto_loss_batch * n_proto
                total_prototypes += n_proto
                n_samples += current_batch_size

                metric_evaluator.update(
                    original_labels,
                    self.rescale_targets(targets, test_id),
                    self.tag_probabilities(outputs),
                    self.taw_probabilities(outputs, test_id),
                )

            taw_acc, tag_acc = metric_evaluator.get(verbose=verbose)
            classification_loss = classification_loss / n_samples

            if current_training_task > 0:
                ceos_loss = ceos_loss / n_samples
                prototype_loss = prototype_loss / total_prototypes
                overall_loss = classification_loss + ceos_loss + prototype_loss
            else:
                overall_loss = classification_loss

            if verbose:
                print(f" - classification loss: {classification_loss}")
                if current_training_task > 0:
                    print(f" - ceos loss: {ceos_loss}")
                    print(
                        f" - proto loss: {prototype_loss}, N proto {total_prototypes}, bs proto {n_proto}"
                    )

            return taw_acc, tag_acc, overall_loss

    def post_train(self, task_id, trn_loader):
        with torch.no_grad():
            if task_id > 0 and self.prototype_update_sigma != -1:
                print("Final Computing Update Proto")
                start = time()
                new_features, old_features = get_old_new_features(
                    self.model,
                    self.old_model,
                    trn_loader,
                    self.device,
                )
                drift = self.compute_drift(new_features, old_features, device=self.device)
                end = time()
                print(f"Elapsed time {end - start:.3f}s")

                for i, (prototype, variance, proto_label) in enumerate(
                    zip(
                        self.proto_generator.prototype,
                        self.proto_generator.running_proto_variance,
                        self.proto_generator.class_label,
                    )
                ):
                    prototype = prototype.to(self.device)
                    variance = variance.to(self.device)
                    updated_mean = prototype + drift[i]
                    self.proto_generator.update_gaussian(proto_label, updated_mean, variance)
                    self.proto_generator.prototype[i] = updated_mean

                self.proto_generator.running_proto = deepcopy(self.proto_generator.prototype)

            efm_matrix = EmpiricalFeatureMatrix(self.device, out_path=self.out_path)
            efm_matrix.compute(self.model, deepcopy(trn_loader), task_id)
            self.previous_efm = efm_matrix.get()
            matrix_rank = torch.linalg.matrix_rank(self.previous_efm)
            print(f"Computed Matrix Rank {matrix_rank}")

        save_efm(self.previous_efm, task_id, self.out_path)

        print("Computing New Task Prototypes")
        self.proto_generator.compute(self.model, deepcopy(trn_loader), task_id)

    def compute_drift(self, new_features, old_features, device):
        new_features = new_features.to(device)
        old_features = old_features.to(device)
        feature_drift = new_features - old_features
        running_prototypes = torch.stack(self.proto_generator.running_proto, dim=0).to(device)
        efm = self.previous_efm.to(device)

        num_prototypes, num_samples = running_prototypes.shape[0], new_features.shape[0]
        distance = torch.zeros(num_prototypes, num_samples, device=device)

        for i in range(num_prototypes):
            diff = (old_features - running_prototypes[i]).unsqueeze(1)
            efm_expanded = efm.unsqueeze(0).expand(num_samples, -1, -1)
            score = -torch.bmm(torch.bmm(diff, efm_expanded), diff.permute(0, 2, 1))
            distance[i] = score.flatten()

        min_distance, max_distance = distance.min(), distance.max()
        scaled_distance = (distance - min_distance) / (max_distance - min_distance)

        sigma2 = 2 * (self.prototype_update_sigma ** 2)
        weights = torch.exp(scaled_distance / sigma2)
        normalized_weights = weights / weights.sum(dim=1, keepdim=True)

        displacement = torch.zeros(num_prototypes, feature_drift.size(1), device=device)
        for i in range(num_prototypes):
            displacement[i] = (normalized_weights[i].unsqueeze(1) * feature_drift).sum(dim=0)

        return displacement

    def feature_cutmix_batch(
        self,
        feat_a: torch.Tensor,
        lab_a: torch.Tensor,
        feat_b: torch.Tensor,
        lab_b: torch.Tensor,
        alpha: float = 1.0,
        contiguous: bool = False,
    ):
        assert feat_a.shape == feat_b.shape
        batch_size, feat_dim = feat_a.shape
        device = feat_a.device

        lam = np.random.beta(alpha, alpha, size=batch_size).astype("float32")
        lam_tensor = torch.from_numpy(lam).to(device)

        masks = torch.zeros(batch_size, feat_dim, device=device, dtype=feat_a.dtype)
        for i, keep_ratio in enumerate(lam):
            replaced_dims = max(1, int(round((1.0 - keep_ratio) * feat_dim)))
            if contiguous:
                start = random.randint(0, feat_dim - replaced_dims)
                masks[i, start:start + replaced_dims] = 1.0
            else:
                idx = torch.randperm(feat_dim, device=device)[:replaced_dims]
                masks[i, idx] = 1.0

        mixed_feat = feat_a * (1.0 - masks) + feat_b * masks
        return mixed_feat, lab_a, lab_b, lam_tensor

    def expansive_oversampling(
        self,
        proto_aug: torch.Tensor,
        proto_aug_label: torch.Tensor,
        batch_features: torch.Tensor,
        batch_labels: torch.Tensor,
        k: int = 1,
        normalize: bool = True,
    ):
        device = batch_features.device
        proto_aug = proto_aug.to(device)
        proto_aug_label = proto_aug_label.to(device)
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        num_prototypes, feat_dim = proto_aug.shape
        if num_prototypes == 0 or batch_features.size(0) == 0:
            return (
                torch.empty((0, feat_dim), device=device, dtype=batch_features.dtype),
                torch.empty((0,), device=device, dtype=torch.long),
            )

        dist = torch.cdist(proto_aug, batch_features, p=2)
        enemy_mask = proto_aug_label.unsqueeze(1).ne(batch_labels)
        dist[~enemy_mask] = float("inf")

        has_enemy = enemy_mask.any(dim=1)
        proto_aug = proto_aug[has_enemy]
        proto_aug_label = proto_aug_label[has_enemy]
        dist = dist[has_enemy]
        num_prototypes = proto_aug.size(0)

        if num_prototypes == 0:
            return (
                torch.empty((0, feat_dim), device=device, dtype=batch_features.dtype),
                torch.empty((0,), device=device, dtype=torch.long),
            )

        k = int(min(k, dist.size(1)))
        values, indices = torch.topk(dist, k, dim=1, largest=False)
        valid = torch.isfinite(values)

        proto_indices = torch.arange(num_prototypes, device=device).unsqueeze(1).expand(num_prototypes, k)
        proto_indices = proto_indices[valid]
        enemy_indices = indices[valid]

        if proto_indices.numel() == 0:
            return (
                torch.empty((0, feat_dim), device=device, dtype=batch_features.dtype),
                torch.empty((0,), device=device, dtype=torch.long),
            )

        proto_feat = proto_aug[proto_indices]
        enemy_feat = batch_features[enemy_indices]
        proto_label = proto_aug_label[proto_indices]

        lam = 0.7 + 0.3 * torch.rand(proto_feat.size(0), 1, device=device)
        mixed_features = lam * proto_feat + (1.0 - lam) * enemy_feat
        if normalize:
            mixed_features = F.normalize(mixed_features, p=2, dim=1)

        return mixed_features, proto_label


def make_samples_per_class(
    class_per_task: int,
    task_id: int,
    current_samples: int = 500,
    old_samples: int = 1,
):
    total_seen = (task_id + 1) * class_per_task
    samples_per_class = [old_samples] * total_seen
    start_idx = task_id * class_per_task
    end_idx = start_idx + class_per_task
    for i in range(start_idx, end_idx):
        samples_per_class[i] = current_samples
    return samples_per_class



def make_balanced_samples_per_class(
    class_per_task: int,
    task_id: int,
    current_samples: int = 500,
    min_old_samples: int = 1,
):
    current_class_count = class_per_task
    old_class_count = task_id * class_per_task

    if old_class_count == 0:
        old_class_weight = 0
    else:
        old_class_weight = (current_samples * current_class_count) // old_class_count
        old_class_weight = max(old_class_weight, min_old_samples)

    total_seen = (task_id + 1) * class_per_task
    samples_per_class = [old_class_weight] * total_seen

    start = task_id * class_per_task
    end = start + class_per_task
    for i in range(start, end):
        samples_per_class[i] = current_samples

    return samples_per_class



def make_incrementing_samples_per_class(
    class_per_task: int,
    task_id: int,
    total_tasks: int,
    current_samples: int = 500,
    min_old_samples: int = 1,
):
    assert total_tasks > 1, "Need at least two tasks for an incrementing schedule"

    progress = task_id / (total_tasks - 1)
    old_class_weight = int(
        min_old_samples + progress * (current_samples - min_old_samples)
    )

    total_seen = (task_id + 1) * class_per_task
    samples_per_class = [old_class_weight] * total_seen

    start = task_id * class_per_task
    end = start + class_per_task
    for i in range(start, end):
        samples_per_class[i] = current_samples

    return samples_per_class



def random_match_proto(
    proto_feat: torch.Tensor,
    proto_label: torch.Tensor,
    full_feat: torch.Tensor,
    full_label: torch.Tensor,
):
    batch_size, _ = proto_feat.shape
    full_batch_size, _ = full_feat.shape
    device = proto_feat.device

    if full_batch_size >= batch_size:
        idx = torch.randperm(full_batch_size, device=device)[:batch_size]
    else:
        idx = torch.randint(0, full_batch_size, size=(batch_size,), device=device)

    selected_feat = full_feat[idx]
    selected_label = full_label[idx]

    perm = torch.randperm(batch_size, device=device)
    proto_feat = proto_feat[perm]
    proto_label = proto_label[perm]
    selected_feat = selected_feat[perm]
    selected_label = selected_label[perm]

    return proto_feat, proto_label, selected_feat, selected_label
