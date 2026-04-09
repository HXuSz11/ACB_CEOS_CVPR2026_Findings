import abc
import torch
import numpy as np
from copy import deepcopy
from torch.distributions import MultivariateNormal
import torch.nn.functional as F


class ProtoManager(metaclass=abc.ABCMeta):
    def __init__(self, device, task_dict, batch_size, feature_space_size) -> None:
        self.device = device
        self.task_dict = task_dict
        self.batch_size = batch_size
        self.feature_space_size = feature_space_size
        self.prototype = []
        self.variances = []
        self.class_label = []

    @abc.abstractmethod
    def compute(self, model, loader, current_task):
        pass

    @abc.abstractmethod
    def perturbe(self, *args):
        pass

    @abc.abstractmethod
    def update(self, *args):
        pass


class ProtoGenerator(ProtoManager):
    def __init__(self, device, task_dict, batch_size, out_path, feature_space_size) -> None:
        super(ProtoGenerator, self).__init__(device, task_dict, batch_size, feature_space_size)
        self.R = None
        self.running_proto = None
        self.running_proto_variance = []
        self.rank = None
        self.out_path = out_path
        self.current_mean = None
        self.current_std = None
        self.gaussians = {}
        self.rank_list = []

    def compute(self, model, loader, current_task):
        model.eval()
        features_list = []
        label_list = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.long().to(self.device)
                _, features = model(images)

                label_list.append(labels)
                features_list.append(features)

        label_list = torch.cat(label_list, dim=0)  # on device
        features_list = torch.cat(features_list, dim=0)  # on device

        for label in self.task_dict[current_task]:
            mask = (label_list == label)
            feature_classwise = features_list[mask]  # on device

            proto = feature_classwise.mean(dim=0)  # on device
            # covariance on device
            covariance = torch.cov(feature_classwise.T)

            self.running_proto_variance.append(covariance)
            self.prototype.append(proto)
            self.class_label.append(int(label))

            eye = torch.eye(covariance.size(0), device=self.device)
            self.gaussians[label] = MultivariateNormal(
                proto,
                covariance_matrix=covariance + 1e-5 * eye,
            )

        self.running_proto = deepcopy(self.prototype)

    def update_gaussian(self, proto_label, mean, var):
        mean = mean.to(self.device)
        var = var.to(self.device)
        eye = torch.eye(var.size(0), device=self.device)
        self.gaussians[proto_label] = MultivariateNormal(
            mean,
            covariance_matrix=var + 1e-5 * eye,
        )

    def perturbe(self, current_task, protobatchsize=64):
        # number of classes seen before
        num_old = sum(len(self.task_dict[i]) for i in range(current_task))
        # random indices on device
        idx = torch.randperm(num_old, device=self.device)
        labels_tensor = torch.tensor(self.class_label, dtype=torch.long, device=self.device)
        proto_aug_label = labels_tensor[idx]

        # adjust to batch size
        if proto_aug_label.size(0) < protobatchsize:
            to_add = protobatchsize - proto_aug_label.size(0)
            repeat_labels = proto_aug_label.repeat(int(np.ceil(to_add / proto_aug_label.size(0))))
            proto_aug_label = torch.cat([proto_aug_label, repeat_labels[:to_add]], dim=0)
        else:
            proto_aug_label = proto_aug_label[:protobatchsize]

        proto_aug_label, _ = torch.sort(proto_aug_label)
        num_classes = len(self.class_label)
        one_hot = F.one_hot(proto_aug_label, num_classes=num_classes).sum(dim=0)

        proto_aug_list = []
        for class_idx, n_samples in enumerate(one_hot):
            if n_samples > 0:
                samples = self.gaussians[class_idx].sample((int(n_samples),))
                proto_aug_list.append(samples)

        proto_aug = torch.cat(proto_aug_list, dim=0)
        n_proto = proto_aug.size(0)

        # shuffle
        shuffle_idx = torch.randperm(n_proto, device=self.device)
        proto_aug = proto_aug[shuffle_idx]
        proto_aug_label = proto_aug_label[shuffle_idx]

        return proto_aug, proto_aug_label, n_proto
    


    def update(self, *args):
        pass
