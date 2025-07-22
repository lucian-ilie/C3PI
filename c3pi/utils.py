import pandas as pd
import torch
from typing import Union
from pathlib import Path
import math
from typing import List, Tuple
import pandas as pd
import torch
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
)
import os


def average_predictions_by_group(
    input_csv: Union[str, Path],
    prediction_tensor_path: Union[str, Path],
    output_csv: Union[str, Path],
    group_size: int = 8
) -> None:
    """
    Reads a CSV file and a PyTorch tensor file of predictions, averages predictions
    over fixed-size groups, and writes the results to a new CSV.

    Args:
        input_csv (str or Path): Path to the input CSV file.
        prediction_tensor_path (str or Path): Path to the tensor file with predictions.
        output_csv (str or Path): Path to the output CSV file.
        group_size (int): Number of entries to average over. Default is 8.
    """
    df = pd.read_csv(input_csv, sep='\t', header=None, names=['proteinA', 'proteinB', 'label'])

    temp = torch.load(prediction_tensor_path).squeeze().numpy()
    averaged_preds = [sum(temp[i:i+group_size]) / group_size for i in range(0, len(temp), group_size)]

    if len(averaged_preds) != len(df):
        raise ValueError(f"Length mismatch: {len(averaged_preds)} predictions vs {len(df)} rows in CSV.")

    df['prediction'] = averaged_preds
    df.to_csv(output_csv, sep='\t', index=None)
    print(f"Saved updated predictions to {output_csv}")




def extended_evaluation_vthr(
    true_labels: List[int],
    predictions: List[float],
    threshold: float
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Evaluates a binary classification model's performance at a given threshold.

    Args:
        true_labels (List[int]): Ground truth binary labels.
        predictions (List[float]): Model's prediction probabilities for the positive class.
        threshold (float): Threshold to convert probabilities to binary predictions.

    Returns:
        Tuple containing:
        - accuracy
        - sensitivity (recall)
        - specificity
        - precision
        - MCC
        - F1 score
        - AUROC
        - AUPR
    """
    pred_labels = [1 if p > threshold else 0 for p in predictions]

    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = sensitivity
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    mcc_denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    fpr, tpr, _ = roc_curve(true_labels, predictions)
    auroc = auc(fpr, tpr)

    pr_precision, pr_recall, _ = precision_recall_curve(true_labels, predictions)
    aupr = auc(pr_recall, pr_precision)

    return accuracy, sensitivity, specificity, precision, mcc, f1, auroc, aupr


def evaluate_models_varying_thresholds(
    species: List[str],
    models: List[str],
    results_dir: str,
    result_file: str = "results/extendedEval.tsv",
    prediction_file: str = "preds_8_full_ens.tsv",
    threshold_range: Tuple[int, int] = (40, 55)
) -> None:
    """
    Evaluates each model on each species for a range of thresholds and writes results to a file.

    Args:
        species (List[str]): List of species names.
        models (List[str]): List of model identifiers.
        results_dir (str): Base directory containing predictions.
        result_file (str): Output file for saving evaluation results.
        prediction_file (str): Filename of the predictions within results_dir.
        threshold_range (Tuple[int, int]): Range of thresholds (in integer percent) to evaluate.
    """
    output_path = Path(result_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a") as w:
        w.write("model\tkind\tthr\taccuracy\tprecision\trecall\tAUROC\tAUPR\tF1\n")
        for kind in species:
            for model in models:
                for thr_percent in range(*threshold_range):
                    thr = thr_percent / 100.0
                    pred_path = prediction_file
                    df = pd.read_csv(pred_path, sep='\t')
                    acc, sens, spec, prec, mcc, f1, auroc, aupr = extended_evaluation_vthr(
                        df["label"].tolist(), df["prediction"].tolist(), thr
                    )
                    w.write(f"{model}\t{kind}\t{thr:.2f}\t{acc:.4f}\t{prec:.4f}\t{sens:.4f}\t{auroc:.4f}\t{aupr:.4f}\t{f1:.4f}\n")
            w.write("\n")

def removeDeficientEmbedding(fastaAddress, embdAddress, expectedEmbdLength):
    '''
    Remove deficient embeddings caused by time limits in sbatch.

    Parameters:
    - fastaAddress (str): The file path to the input FASTA file containing protein sequences.
    - embdAddress (str): The directory path where the embeddings are stored.
    - expectedEmbdLength (int): The expected length of each embedding.

    Returns:
    None

    This function reads protein sequences from the specified FASTA file and checks the corresponding
    embedding files in the provided directory. If the length of the embedding does not match the
    expected length, or if any errors occur during processing, the corresponding embedding file is removed.

    Usage:
    removeDeficientEmbedding("dataset/seq/Pruned_ecoli.fasta", "dataset/embd/ecoli", expectedEmbdLength)
    '''
    cnt = 0
    f = open(fastaAddress, 'r')
    while True:
        if cnt % 1000 == 0:
            print(cnt)
        cnt +=1
        pid = f.readline()
        pseq = f.readline()
        if pseq == "":
            break
        try:
            num_lines = 0
            for line in open('{}/{}.embd'.format(embdAddress,pid.strip()[1:])):
                num_lines += 1
                try:
                    embdLength = len(line.strip().split(':')[1].split(' '))
                except:
                    os.remove('{}/{}.embd'.format(embdAddress,pid.strip()[1:]))
                    break
                if embdLength != expectedEmbdLength:
                    #print("pid: {}, len seq: {}, num_lines: {}".format(pid.strip()[1:], len(pseq.strip()), embdLength))
                    os.remove('{}/{}.embd'.format(embdAddress,pid.strip()[1:]))
            if len(pseq.strip())+1 != num_lines:
                if os.path.exists('{}/{}.embd'.format(embdAddress,pid.strip()[1:])):
                    os.remove('{}/{}.embd'.format(embdAddress,pid.strip()[1:]))
                #print("pid: {}, len seq: {}, num_lines: {}".format(pid.strip()[1:], len(pseq.strip()), num_lines))
        except Exception as e:
            #print(e)
            pass

removeDeficientEmbedding("/home/mohsenh/projects/def-ilie/mohsenh/ppi/dataset/seq/human_swissprot_oneliner.fasta",'/home/mohsenh/projects/def-ilie/mohsenh/c3pi/ankhEmbd', 1536)