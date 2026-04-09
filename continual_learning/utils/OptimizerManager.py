import torch
import inspect


# --------------------------------------------------------------------------
# Drop-in helper for creating MultiStepLR no matter which PyTorch version
# --------------------------------------------------------------------------
def _make_multistep_lr(optimizer, *, milestones, gamma=0.1, verbose=True):
    """
    Create a torch.optim.lr_scheduler.MultiStepLR even if the underlying
    PyTorch build lacks the `verbose` constructor argument.

    Parameters
    ----------
    optimizer   : torch.optim.Optimizer
    milestones  : list[int] – epochs where the LR decays
    gamma       : float – decay factor (default 0.1)
    verbose     : bool  – print LR changes when supported (default True)

    Returns
    -------
    lr_scheduler.MultiStepLR
    """
    sig = inspect.signature(torch.optim.lr_scheduler.MultiStepLR)
    if "verbose" in sig.parameters:
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma, verbose=verbose
        )
    # Older signature – silently drop `verbose`
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=gamma
    )


# --------------------------------------------------------------------------
# OptimizerManager
# --------------------------------------------------------------------------
class OptimizerManager:
    def __init__(self, approach, dataset, rotation, args) -> None:
        self.approach = approach
        self.dataset = dataset
        self.rotation = rotation
        self.args = args
        self.scale = args.lr_scale  # Scale factor for batch size, if needed

    # ----------------------------------------------------------------------
    # Build optimizer & LR scheduler
    # ----------------------------------------------------------------------
    def get_optimizer(self, task_id, model, auxiliary_classifier):
        # ------------------------------------------------------------------
        # ─── FIRST TASK (usually much larger training budget) ──────────────
        # ------------------------------------------------------------------
        if task_id == 0:
            params_to_optimize = (
                [p for p in model.backbone.parameters() if p.requires_grad]
                + [p for p in model.heads.parameters() if p.requires_grad]
            )

            # Optional self-rotation head
            if self.rotation:
                params_to_optimize += [
                    p for p in auxiliary_classifier.parameters() if p.requires_grad
                ]
                print("Optimizing Self Rotation")

            # -------- ImageNet (or large-scale) settings -------------------
            if self.dataset in {"imagenet-subset", "imagenet-1k"}:
                lr_first_task = 0.1 * self.scale  # Scale LR for larger batch size
                gamma = 0.1
                custom_weight_decay = 5e-4
                custom_momentum = 0.9
                milestones_first_task = [80, 120, 150]
                # milestones_first_task = [
                #     round(0.50 * self.args.epochs_first_task),
                #     round(0.75 * self.args.epochs_first_task),
                #     round(0.94 * self.args.epochs_first_task),
                # ]

                optimizer = torch.optim.SGD(
                    params_to_optimize,
                    lr=lr_first_task,
                    momentum=custom_momentum,
                    weight_decay=custom_weight_decay,
                )
                scheduler = _make_multistep_lr(
                    optimizer,
                    milestones=milestones_first_task,
                    gamma=gamma,
                    verbose=True,
                )

                print(
                    "Using SGD Optimizer:\n"
                    f"  LR={lr_first_task}, gamma={gamma}, "
                    f"milestones={milestones_first_task}, "
                    f"weight_decay={custom_weight_decay}, "
                    f"momentum={custom_momentum}"
                )

            # -------- CIFAR/TinyImageNet etc. -----------------------------
            else:
                lr_first_task = 1e-3 * self.scale  # Scale LR for larger batch size
                # milestones_first_task = [45, 90]
                milestones_first_task = [
                    round(0.45 * self.args.epochs_first_task),
                    round(0.90 * self.args.epochs_first_task)
                ]
                custom_weight_decay = 2e-4
                gamma = 0.1

                optimizer = torch.optim.AdamW(
                    params_to_optimize, lr=lr_first_task, weight_decay=custom_weight_decay
                )
                scheduler = _make_multistep_lr(
                    optimizer,
                    milestones=milestones_first_task,
                    gamma=gamma,
                    verbose=True,
                )

                print(
                    "Using Adam Optimizer:\n"
                    f"  LR={lr_first_task}, gamma={gamma}, "
                    f"milestones={milestones_first_task}, "
                    f"weight_decay={custom_weight_decay}"
                )

            return optimizer, scheduler

        # ------------------------------------------------------------------
        # ─── SUBSEQUENT TASKS (fine-tuning / incremental) ─────────────────
        # ------------------------------------------------------------------
        model.freeze_bn()  # keep BatchNorm statistics fixed

        # ---------- ImageNet (large-scale) fine-tuning --------------------
        if self.dataset in {"imagenet-subset", "imagenet-1k"}:
            backbone_lr = 1e-5 * self.scale  # Scale LR for larger batch size
            head_lr = 1e-4 * self.scale  # Scale LR for larger batch size
            custom_weight_decay = 2e-4

            backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
            old_head_params = [
                p for p in model.heads[:-1].parameters() if p.requires_grad
            ]
            new_head_params = [
                p for p in model.heads[-1].parameters() if p.requires_grad
            ]
            head_params = old_head_params + new_head_params

            optimizer = torch.optim.Adam(
                [
                    {"params": head_params, "lr": head_lr},
                    {"params": backbone_params},  # uses backbone_lr
                ],
                lr=backbone_lr,
                weight_decay=custom_weight_decay,
            )

            print(
                "Using Adam Optimizer (fixed LR):\n"
                f"  Backbone LR={backbone_lr}, Head LR={head_lr}, "
                f"weight_decay={custom_weight_decay}"
            )

        # ---------- Smaller datasets -------------------------------------
        else:
            old_head_params = [
                p for p in model.heads[:-1].parameters() if p.requires_grad
            ]
            new_head_params = [
                p for p in model.heads[-1].parameters() if p.requires_grad
            ]
            head_params = old_head_params + new_head_params

            params_to_optimize = (
                [p for p in model.backbone.parameters() if p.requires_grad] + head_params
            )

            backbone_lr = head_lr = 1e-4 * self.scale  # Scale LR for larger batch size
            custom_weight_decay = 2e-4

            optimizer = torch.optim.AdamW(
                params_to_optimize, lr=backbone_lr, weight_decay=custom_weight_decay
            )

            print(
                "Using Adam Optimizer (fixed LR):\n"
                f"  Backbone LR={backbone_lr}, Head LR={head_lr}, "
                f"weight_decay={custom_weight_decay}"
            )

        # Same LR for the entire incremental phase – decay far in the future
        scheduler = _make_multistep_lr(
            optimizer, milestones=[1000, 2000], gamma=0.1, verbose=True
        )
        return optimizer, scheduler
# import torch 


# class OptimizerManager:
    
#     def __init__(self, approach, dataset, rotation) -> None:
        
#         self.approach = approach
#         self.dataset = dataset
#         self.rotation = rotation
    
#     def get_optimizer(self, task_id, model, auxiliary_classifier):
#         ## Large First Task Training
#         if task_id == 0:
#             params_to_optimize = [p for p in  model.backbone.parameters() if p.requires_grad] + [p for p in model.heads.parameters() if p.requires_grad]
           
#             if self.rotation:
#                 params_to_optimize += [p for p in auxiliary_classifier.parameters() if p.requires_grad]
#                 print("Optimizing Self Rotation")
       
                
                
#             if self.dataset == "imagenet-subset" or self.dataset == "imagenet-1k":
                
#                 lr_first_task = 0.1 
#                 gamma = 0.1 
#                 custom_weight_decay = 5e-4 
#                 custom_momentum = 0.9  
                
#                 milestones_first_task = [80, 120, 150]
#                 optimizer = torch.optim.SGD(params_to_optimize, lr=lr_first_task, momentum=custom_momentum,
#                                                 weight_decay=custom_weight_decay)
                
#                 reduce_lr_on_plateau = torch.optim.lr_scheduler.MultiStepLR(optimizer,
#                                                             milestones=milestones_first_task,
#                                                             gamma=gamma, verbose=True
#                                                             )
#                 print("Using SGD Optimizer With PASS setting: \n\
#                         LR: {}, Step Decay {} with Milestones {}, Weight Decay {}, Momentum {} ".format(lr_first_task,
#                                                                                                             gamma,
#                                                                                                             milestones_first_task,
#                                                                                                             custom_weight_decay,
#                                                                                                             custom_momentum,
#                                                                                                             ))
#             else:

#                 lr_first_task = 1e-3
#                 milestones_first_task = [45, 90]
#                 custom_weight_decay = 2e-4
#                 gamma = 0.1
#                 optimizer = torch.optim.Adam(params_to_optimize, lr=lr_first_task, weight_decay=custom_weight_decay)
#                 reduce_lr_on_plateau = torch.optim.lr_scheduler.MultiStepLR(optimizer,
#                                                                                 milestones=milestones_first_task,
#                                                                             gamma=gamma, verbose=True)
#                 print("Using Adam Optimizer With PASS setting: \n\
#                         LR: {}, Step Decay {} with Milestones {}, Weight Decay {} ".format(lr_first_task,
#                                                                                             gamma,
#                                                                                             milestones_first_task,
#                                                                                             custom_weight_decay
#                                                                                                             )) 
            
#             return optimizer, reduce_lr_on_plateau 
        
#         else:
#             # Oprimization Settings Next Tasks
#             model.freeze_bn()
#             if self.dataset == "imagenet-subset"  or self.dataset == "imagenet-1k": 
#                 backbone_lr = 1e-5
#                 head_lr = 1e-4
#                 custom_weight_decay = 2e-4
#                 backbone_params = [p for p in  model.backbone.parameters() if p.requires_grad]
            
#                 old_head_params = [p for p in model.heads[:-1].parameters()  if p.requires_grad]
                
#                 new_head_params = [p for p in  model.heads[-1].parameters() if p.requires_grad]
#                 head_params = old_head_params + new_head_params
                
#                 optimizer = torch.optim.Adam([{'params': head_params, 'lr':head_lr},
#                                 {'params': backbone_params}
#                                     ],lr=backbone_lr, 
#                                     weight_decay=custom_weight_decay)
#                 print("Using Adam Optimizer With Fixed LR: \n\
#                         Backbone LR: {}, Head LR {}, Weight Decay {},".format(backbone_lr,
#                                                                                 head_lr,      
#                                                                                 custom_weight_decay)) 

#             else:
                
#                 old_head_params = [p for p in model.heads[:-1].parameters()  if p.requires_grad]

#                 new_head_params = [p for p in  model.heads[-1].parameters() if p.requires_grad]
#                 head_params = old_head_params + new_head_params

#                 params_to_optimize = [p for p in model.backbone.parameters() if p.requires_grad] +  head_params 
#                 backbone_lr = head_lr = 1e-4
#                 custom_weight_decay = 2e-4
#                 optimizer =  torch.optim.Adam(params_to_optimize, lr=backbone_lr, weight_decay=2e-4)
#                 print("Using Adam Optimizer With Fixed LR: \n\
#                         Backbone LR: {}, Head LR {}, Weight Decay {}".format(backbone_lr,
#                                                                             head_lr,      
#                                                                             custom_weight_decay)) 

#             # Fixed Scheduler
#             reduce_lr_on_plateau = torch.optim.lr_scheduler.MultiStepLR(optimizer,
#                                                             milestones=[1000,2000],
#                                                             gamma=0.1, verbose=True
#                                                             )
#             return optimizer, reduce_lr_on_plateau 
            
            
            

  


    
 