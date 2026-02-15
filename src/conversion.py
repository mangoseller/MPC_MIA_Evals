"""
Plaintext → MPC model conversion via direct weight transfer.

Instead of training MPC models from scratch (which accumulates approximation
error from encrypted arithmetic over many epochs), we train in plaintext and
transfer the learned weights to structurally-identical MPC models. This ensures
both models have identical parameters, so any difference in MIA susceptibility
is attributable to MPC's computation rather than different model quality.

Weight transfer works because both model families use the same attribute names
for their parameterized layers (conv1, conv2, fc1, fc2, bn1-bn4, etc.), so
the state_dict keys for Conv2d, Linear, and BatchNorm layers match exactly.
"""

import torch as t
import crypten
import crypten.nn as cnn
from models import PLAINTEXT_TO_MPC, mpc_name_from_plaintext


# Layers that carry learnable parameters or running statistics
_PARAMETERIZED_TYPES_PT = (t.nn.Conv2d, t.nn.Linear, t.nn.BatchNorm1d, t.nn.BatchNorm2d)
_PARAMETERIZED_TYPES_MPC = (cnn.Conv2d, cnn.Linear, cnn.BatchNorm1d, cnn.BatchNorm2d)


def _transfer_layer(src, dst):
    """Copy weight, bias, and BatchNorm buffers from a PyTorch layer to a CrypTen layer."""
    if hasattr(src, 'weight') and hasattr(dst, 'weight'):
        dst.weight.data.copy_(src.weight.data)
    if hasattr(src, 'bias') and src.bias is not None and hasattr(dst, 'bias') and dst.bias is not None:
        dst.bias.data.copy_(src.bias.data)
    # BatchNorm running statistics and tracking counter
    for attr in ('running_mean', 'running_var', 'num_batches_tracked'):
        if hasattr(src, attr) and hasattr(dst, attr):
            src_buf = getattr(src, attr)
            dst_buf = getattr(dst, attr)
            if src_buf is not None and dst_buf is not None:
                dst_buf.data.copy_(src_buf.data)


def _get_parameterized_layers(model, types):
    """
    Collect parameterized layers by attribute name (skipping the Sequential
    to avoid duplicates, since self.network contains references to the same objects).
    """
    layers = {}
    for name, module in model._modules.items():
        if name == 'network':
            continue  # Skip Sequential — layers are also registered as direct attributes
        if isinstance(module, types):
            layers[name] = module
    return layers


def convert_plaintext_to_mpc(plaintext_model, plaintext_name, num_classes=10, verbose=True):
    """
    Create an MPC model with the same architecture and copy all learned
    weights from a trained plaintext model.

    The MPC model is returned in its unencrypted state so it can be
    encrypted later for evaluation.

    Args:
        plaintext_model: Trained PyTorch model (on any device; weights are copied to CPU)
        plaintext_name: Key in PLAINTEXT_MODELS, e.g. 'PlainTextCNN_Sigmoid'
        num_classes: Number of output classes
        verbose: Print transfer summary

    Returns:
        mpc_model: CrypTen model with transferred weights (unencrypted, CPU)
        mpc_name: Name string, e.g. 'MpcCNN_Sigmoid'
    """
    if not crypten.is_initialized():
        crypten.init()

    mpc_name = mpc_name_from_plaintext(plaintext_name)
    mpc_constructor = PLAINTEXT_TO_MPC[plaintext_name]

    # Instantiate unencrypted MPC model
    mpc_model = mpc_constructor(num_classes=num_classes)

    # Remember original device so we can restore after the copy
    original_device = next(plaintext_model.parameters()).device
    plaintext_model_cpu = plaintext_model.cpu()

    src_layers = _get_parameterized_layers(plaintext_model_cpu, _PARAMETERIZED_TYPES_PT)
    dst_layers = _get_parameterized_layers(mpc_model, _PARAMETERIZED_TYPES_MPC)

    transferred = 0
    for name in src_layers:
        if name not in dst_layers:
            if verbose:
                print(f"  [WARN] No matching MPC layer for '{name}', skipping")
            continue
        _transfer_layer(src_layers[name], dst_layers[name])
        transferred += 1

    if verbose:
        print(f"  Transferred {transferred} parameterized layers: "
              f"{sorted(src_layers.keys() & dst_layers.keys())}")

    # Restore plaintext model to its original device (caller still holds a reference)
    plaintext_model.to(original_device)

    return mpc_model, mpc_name


def convert_all_plaintext_to_mpc(plaintext_models, num_classes=10, verbose=True):
    """
    Batch-convert all trained plaintext models to their MPC counterparts.

    Args:
        plaintext_models: Dict of {plaintext_name: trained_model}

    Returns:
        mpc_models: Dict of {mpc_name: mpc_model} (unencrypted)
    """
    mpc_models = {}
    for pt_name, pt_model in plaintext_models.items():
        if verbose:
            print(f"\n[CONVERT] {pt_name} → {mpc_name_from_plaintext(pt_name)}")
        mpc_model, mpc_name = convert_plaintext_to_mpc(
            pt_model, pt_name, num_classes=num_classes, verbose=verbose
        )
        mpc_models[mpc_name] = mpc_model
    return mpc_models
