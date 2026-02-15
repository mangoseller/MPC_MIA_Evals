from config import ExperimentConfig, setup_dirs
from dataset_registry import DatasetSpec, get_dataset_spec, list_datasets, DATASETS
from models import (
    PlainTextCNN, PlainTextMLP, PlainTextLeNet,
    MpcCNN, MpcMLP, MpcLeNet, MpcFlatten, MpcTanh, MpcGELU,
    PLAINTEXT_MODELS, MPC_MODELS, PLAINTEXT_TO_MPC,
    mpc_name_from_plaintext,
)
from training import plaintext_train_model, mpc_train_model, mpc_train_epoch, EarlyStopping
from data import partition_dataset_for_mia, train_shadow_models, regenerate_shadow_indices
from attack import AttackNet, prepare_attack_dataset, train_attack_model
from lira import fit_lira_distributions, evaluate_lira
from conversion import convert_plaintext_to_mpc, convert_all_plaintext_to_mpc
from charts import generate_all_charts
from evaluation import (
    compute_roc_metrics,
    evaluate_accuracy, evaluate_accuracy_mpc,
    evaluate_mia,
    evaluate_all_plaintext, evaluate_all_mpc,
    aggregate_across_seeds,
    print_results_summary, print_aggregated_summary,
    save_results, strip_raw_scores,
)
from checkpointing import (
    check_plaintext_target_exists, check_mpc_target_exists,
    check_shadow_models_exist, check_attack_model_exists,
    find_latest_intermediate_checkpoint,
    load_plaintext_model, load_mpc_model,
    load_shadow_models, load_attack_model,
    get_attack_model_for_architecture,
    extract_epochs_from_checkpoint,
)
from runner import run_single_experiment, run_full_experiment, interactive_config_loop
