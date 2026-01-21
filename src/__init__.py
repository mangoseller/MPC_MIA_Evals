from config import ExperimentConfig, setup_dirs
from models import (
    PlainTextCNN, PlainTextMLP, PlainTextLeNet,
    MpcCNN, MpcMLP, MpcLeNet, MpcFlatten, MpcTanh,
    PLAINTEXT_MODELS, MPC_MODELS
)
from training import plaintext_train_model, mpc_train_model, mpc_train_epoch, EarlyStopping
from data import partition_dataset_for_mia, train_shadow_models, regenerate_shadow_indices
from attack import (
    AttackNet,
    prepare_attack_dataset,
    train_attack_model,
    evaluate_mia_attack,
    evaluate_accuracy_loss,
    evaluate_accuracy_loss_mpc
)
from checkpointing import (
    check_plaintext_target_exists,
    check_mpc_target_exists,
    check_shadow_models_exist,
    check_attack_model_exists,
    load_plaintext_model,
    load_mpc_model,
    load_shadow_models,
    load_attack_model,
    get_attack_model_for_architecture
)
from runner import run_experiment, print_results_summary, interactive_config_loop

__all__ = [
    'ExperimentConfig',
    'setup_dirs',
    
    'PlainTextCNN', 'PlainTextMLP', 'PlainTextLeNet',
    'MpcCNN', 'MpcMLP', 'MpcLeNet', 'MpcFlatten', 'MpcTanh',
    'PLAINTEXT_MODELS', 'MPC_MODELS',
    
    'plaintext_train_model', 'mpc_train_model', 'mpc_train_epoch', 'EarlyStopping',
    
    'partition_dataset_for_mia', 'train_shadow_models', 'regenerate_shadow_indices',
    
    'AttackNet',
    'prepare_attack_dataset',
    'train_attack_model',
    'evaluate_mia_attack',
    'evaluate_accuracy_loss',
    'evaluate_accuracy_loss_mpc',
    
    'check_plaintext_target_exists',
    'check_mpc_target_exists',
    'check_shadow_models_exist',
    'check_attack_model_exists',
    'load_plaintext_model',
    'load_mpc_model',
    'load_shadow_models',
    'load_attack_model',
    'get_attack_model_for_architecture',
    
    'run_experiment',
    'print_results_summary',
    'interactive_config_loop',
]
