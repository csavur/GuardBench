from loguru import logger
from tabulate import tabulate
from tqdm import tqdm
from unified_io import create_path, read_json, write_json

from ..datasets import DATASETS, load_dataset
from ..evaluate import evaluate

metric_mapping = {
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "mcc": "MCC",
    "auprc": "AUPRC",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "g_mean": "G-Mean",
    "fpr": "FPR",
    "fnr": "FNR",
}


def format_score(score: float) -> str:
    return "{:.3f}".format(round(score, 3))


def get_results_table(headers, results) -> None:
    return tabulate(results, headers=headers, tablefmt="github", disable_numparse=True)


def collate_fn(batch):
    id = [x["id"] for x in batch]
    label = [x["label"] for x in batch]
    conversation = [x["conversation"] for x in batch]
    return id, label, conversation


def benchmark(
    moderate: callable,
    model_name: str = "moderator",
    out_dir: str = "results",
    batch_size: int = 1,
    datasets: list = "all",
    metrics: list = None,
    **kwargs,
) -> None:
    """Benchmark the effectiveness of the provided moderation function/model.     Additional keyword arguments are passed to the provided moderation function. For example, you can pass the tokenizer and the model to the moderation function. Check the official repository for examples and tutorials.

    Args:
        moderate (callable): Moderation function. It must have at least one parameter named `conversations`.
        model_name (str, optional): "Name of the moderation model". Defaults to "moderator".
        out_dir (str, optional): Directory for the moderation outputs. Defaults to "results".
        batch_size (int, optional): Batch size. Defaults to 32.
        metrics (list, optional): Metrics used to evaluate results. If None, defaults to `["f1", "recall"]`. Available metrics are: `precision`, `recall`, `f1`, `mcc`, `auprc`, `sensitivity`, `specificity`, `g_mean`, `fpr`, `fnr`, `tn`, `fp`, `fn`, `tp`.
        datasets (list, optional): Datasets selected for evaluation. Defaults to "all".
    """

    # Set datasets if "all"  ---------------------------------------------------
    if datasets == "all":
        datasets = list(DATASETS)
    else:
        if isinstance(datasets, str):
            datasets = [datasets]
        invalid = [d for d in datasets if d not in DATASETS]
        if invalid:
            raise ValueError(f"Unknown dataset(s): {invalid}. Available: {list(DATASETS)}")

    # Set metrics if None  -----------------------------------------------------
    if isinstance(metrics, str):
        metrics = [metrics]
    if metrics is None:
        metrics = ["f1", "recall"]

    for m in metrics:
        if m not in metric_mapping:
            raise ValueError(
                f"Metric `{m}` is not supported. Supported metrics are: `precision`, `recall`, `f1`, `mcc`, `auprc`, `sensitivity`, `specificity`, `g_mean`, `fpr`, `fnr`, `tn`, `fp`, `fn`, `tp`."
            )

    # Benchmarking Effectiveness -----------------------------------------------
    logger.start("Benchmarking Effectiveness")
    results = []
    for i, dataset_name in enumerate(datasets):
        pred_path = create_path(out_dir)
        pred_file = pred_path / dataset_name / f"{model_name}.json"
        true_file = pred_path / dataset_name / "y_true.json"

        # Skip inference if predictions already exist --------------------------
        if pred_file.exists() and true_file.exists():
            logger.info(f"Skipping {dataset_name} (predictions already exist)")
            continue

        try:
            # Dataset ----------------------------------------------------------
            dataset = load_dataset(dataset_name)
            batch_generator = dataset.generate_batches(batch_size)

            # Inference --------------------------------------------------------
            idx = str(i + 1).zfill(2) if len(datasets) > 9 else i + 1
            desc = f"{idx}/{len(datasets)} - {dataset.name}"
            tqdm_kwargs = dict(desc=desc, dynamic_ncols=True, mininterval=1.0)

            ids, y_true, y_pred_prob = [], [], []

            for batch in tqdm(list(batch_generator), **tqdm_kwargs):
                batch_ids, batch_y_true, batch_conversations = collate_fn(batch)
                ids += batch_ids
                y_true += batch_y_true
                y_pred_prob += moderate(conversations=batch_conversations, **kwargs)

            y_true = dict(zip(ids, y_true))
            y_pred_prob = dict(zip(ids, y_pred_prob))

            # Save predictions and labels --------------------------------------
            write_json(y_pred_prob, pred_file)
            write_json(y_true, true_file)

        except Exception as e:
            logger.error(f"Failed on {dataset_name}, skipping: {e}")

    # Evaluate -----------------------------------------------------------------
    pred_path = create_path(out_dir)
    for i, dataset_name in enumerate(datasets):
        try:
            y_true = read_json(pred_path / dataset_name / "y_true.json")
            y_pred_prob = read_json(pred_path / dataset_name / f"{model_name}.json")

            dataset = load_dataset(dataset_name)
            report = evaluate(y_true, y_pred_prob)
            results.append([dataset.name] + [format_score(report[m]) for m in metrics])
        except Exception as e:
            logger.error(f"Failed to evaluate {dataset_name}, skipping: {e}")

    headers = ["Dataset"] + [metric_mapping[m] for m in metrics]
    results_table = get_results_table(headers, results)
    logger.info(f"Results:\n{results_table}")

    logger.success("Done")
