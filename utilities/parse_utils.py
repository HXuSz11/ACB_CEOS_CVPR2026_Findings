import argparse



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--approach", type=str, default="ceos", choices=["ceos"])
    parser.add_argument("--outpath", "-op", default="./", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nw", type=int, default=4, help="num workers for data loader")

    parser.add_argument(
        "--epochs_first_task",
        type=int,
        default=100,
        help="epochs first task, should be changed to 160 for imagenet-subset and imagenet-1k",
    )
    parser.add_argument("--epochs_next_task", type=int, default=100, help="epochs next task")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="batch size of data, should be changed to 256 for imagenet-1k",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lr_scale", type=float, default=1.0, help="scale the learning rate")

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar100", "tiny-imagenet", "imagenet-subset", "imagenet-1k"],
        help="dataset to use",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data",
        help="path where imagenet subset, imagenet-1k, tiny-imagenet are saved",
    )
    parser.add_argument(
        "--n_class_first_task",
        type=int,
        default=50,
        help="if greater than -1 use a larger number of classes for the first task",
    )
    parser.add_argument("--n_task", type=int, default=6, help="number of tasks")
    parser.add_argument(
        "--valid_size",
        type=float,
        default=0.0,
        help="percentage of train for validation set, default is no validation",
    )

    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18"])
    parser.add_argument(
        "--firsttask_modelpath",
        type=str,
        default="None",
        help="path to a pre-trained first-task model",
    )

    parser.add_argument("--ceos_lamb", default=10.0, type=float, help="lambda associated with the EFM")
    parser.add_argument(
        "--ceos_protobatchsize",
        type=int,
        default=64,
        help="prototype batch size, should be changed to 256 for imagenet-1k",
    )
    parser.add_argument("--ceos_damping", type=float, default=0.1, help="damping hyperparameter")
    parser.add_argument(
        "--ceos_protoupdate",
        type=float,
        default=0.2,
        help="prototype-update sigma; use -1 to disable prototype updates",
    )
    parser.add_argument("--focal_loss", action="store_true", help="use focal loss")
    parser.add_argument("--partial", action="store_true", help="use partial focal loss")
    parser.add_argument("--cb_loss", action="store_true", help="use class-balanced loss")
    parser.add_argument("--balanced_weight", action="store_true", help="balance old/new class weights")
    parser.add_argument("--ascending_weight", action="store_true", help="increase old-class weights over time")
    parser.add_argument("--eos", action="store_true", help="use expansive over-sampling")
    parser.add_argument("--eos_k", default=1, type=int, help="number of nearest enemies for EOS")
    parser.add_argument("--proto_lambda", default=1.0, type=float, help="prototype-loss weight")
    parser.add_argument("--cls_lambda", default=1.0, type=float, help="classification-loss weight")
    parser.add_argument("--ceos_lambda", default=1.0, type=float, help="CEOS regularization weight")
    parser.add_argument("--start_old", default=1, type=int, help="base virtual count for old classes")
    parser.add_argument("--noreplay", action="store_true", help="disable replay")

    parser.add_argument("--cutmix", action="store_true", help="use feature-space CutMix")
    parser.add_argument("--cutmix_contiguous", action="store_true", help="use contiguous feature spans in CutMix")
    parser.add_argument("--cutmix_alpha", default=1.0, type=float, help="Beta distribution alpha for CutMix")
    parser.add_argument("--cutmix_lambda", default=1.0, type=float, help="CutMix loss weight")

    args = parser.parse_args()

    non_default_args = {
        opt.dest: getattr(args, opt.dest)
        for opt in parser._option_string_actions.values()
        if hasattr(args, opt.dest) and opt.default != getattr(args, opt.dest)
    }
    default_args = {
        opt.dest: opt.default
        for opt in parser._option_string_actions.values()
        if hasattr(args, opt.dest)
    }
    all_args = vars(args)

    return args, all_args, non_default_args, default_args
