import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, precision_recall_curve, auc
import pandas as pd
import math
import glob

def extendedEvaluation(trueLable, prediction, predictedLabel):
    """
    Evaluate the performance of a binary classification model and print various metrics.

    Parameters:
    - trueLabel (array-like): True labels of the binary classification.
    - prediction (array-like): Predicted probability scores for the positive class.
    - predictedLabel (array-like): Predicted binary labels.

    Returns:
    Tuple containing evaluation metrics:
    - accuracy (float): Accuracy of the model.
    - sensitivity (float): Sensitivity (True Positive Rate/Recall) of the model.
    - specificity (float): Specificity (True Negative Rate) of the model.
    - precision (float): Precision of the model.
    - mcc (float): Matthews correlation coefficient.
    - f1 (float): F1 score of the model.
    - AUROC (float): Area Under the Receiver Operating Characteristic curve.
    - AUPR (float): Area Under the Precision-Recall curve.

    This function computes various classification metrics, prints them, and returns them as a tuple.
    It includes Sensitivity, Specificity, Precision, Matthews correlation coefficient, F1 score, 
    Accuracy, Area Under the ROC curve (AUROC), and Area Under the Precision-Recall curve (AUPR).
    """
    tn, fp, fn, tp = confusion_matrix(trueLable, predictedLabel).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    recall_val = tp / (tp + fn)
    precision_val = tp / (tp + fp)
    mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val)
    accuracy = (tp+tn)/(tp+tn+fp+fn)

    print("sensitivity: ", sensitivity)
    print("Specificity: ", specificity)
    #print("Recall: ", recall_val)
    print("Precision: ", precision_val)
    print("MCC: ", mcc)
    print("F1: ", f1)
    
    fpr, tpr, thresholds = roc_curve(trueLable, prediction)
    AUROC = auc(fpr, tpr)
    print("AUROC:", AUROC)

    precision, recall, thresholds = precision_recall_curve(trueLable, prediction)
    AUPR = auc(recall, precision)
    print("AUPR:", AUPR)

    return accuracy, sensitivity, specificity, precision_val, mcc, f1, AUROC, AUPR
    

def model_species_selection(species, models, baseAddress):
    """
    Evaluate and compare different models for each species and save the results to a file.

    Parameters:
    - species (list): List of species names.
    - models (list): List of model names.
    - baseAddress (str): The base directory path where model results are stored.

    Returns:
    None

    This function evaluates and compares different models for each species based on their
    performance metrics such as accuracy, sensitivity, specificity, precision, MCC, F1 score,
    AUROC, and AUPR. The results are appended to an output file in tab-separated format.

    The file 'results/extendedEval.tsv' is created (or appended to) with a header line and
    subsequent lines containing the evaluation metrics for each combination of model and species.
    """
    w=open('results/extendedEval.tsv','a')
    w.write("model\tkind\taccuracy\tsensitivity\tspecificity\tprecision\tmcc\tf1\tAUROC\tAUPR\n")
    for kind in species:
        for model in models:        
            df = pd.read_csv("{}/{}/{}.tsv".format(baseAddress,kind,model), sep='\t')
            accuracy, sensitivity, specificity, precision_val, mcc, f1, AUROC, AUPR = extendedEvaluation(df.label, df.prediction ,df.predictionLabel)
            w.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(model, kind, accuracy, 
                                                              sensitivity, specificity, precision_val, 
                                                              mcc, f1, AUROC, AUPR))
        w.write('\n')
    





def Evaulation(resultAddress):
    """
    Evaluate model performance using metrics such as precision, recall, AUROC, and AUPR.

    Parameters:
    - resultAddress (str): The file path to the result file containing model predictions.

    Returns:
    None

    This function calculates and prints precision, recall, AUROC, and AUPR scores based on the
    provided result file. The results are also appended to an output file 'results/eval.tsv'.

    The function assumes that the result file is in tab-separated format with columns 'label',
    'prediction', and 'predictionLabel'. It extracts the species and model information from the
    file path and prints these details along with the evaluation metrics.
    """
    species = resultAddress.split('/')[-1].split("_")[0]
    model = resultAddress.split('/')[1].split(".")[0][len(species)+1:]
    print(species)
    print(model)
    df = pd.read_csv(resultAddress, sep='\t')
    precision_value = precision_score(df.label, df.predictionLabel)
    print("precision:", precision_value)
    recall_value = recall_score(df.label, df.predictionLabel)
    print("recall:", recall_value)

    fpr, tpr, thresholds = roc_curve(df.label, df.prediction)
    AUROC = auc(fpr, tpr)
    print("AUROC:", AUROC)

    precision, recall, thresholds = precision_recall_curve(df.label, df.prediction)
    AUPR = auc(recall, precision)
    print("AUPR:", AUPR)
    w=open('results/eval.tsv','a')
    w.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(model, species, precision_value, recall_value, AUROC, AUPR))


def surveyEvaulation(resultAddress):
    """
    Evaluate model performance using metrics such as precision, recall, AUROC, and AUPR.

    Parameters:
    - resultAddress (str): The file path to the result file containing model predictions.

    Returns:
    None

    This function calculates and prints precision, recall, AUROC, and AUPR scores based on the
    provided result file. The results are also appended to an output file 'results/eval.tsv'.

    The function assumes that the result file is in tab-separated format with columns 'label',
    'prediction', and 'predictionLabel'. It extracts the species and model information from the
    file path and prints these details along with the evaluation metrics.
    """
    species = resultAddress.split('/')[-1].split("_")[0]
    model = resultAddress.split('/')[1].split(".")[0][len(species)+1:]
    print(species)
    print(model)
    df = pd.read_csv(resultAddress, sep='\t')
    precision_value = precision_score(df.label, df.predictionLabel)
    print("precision:", precision_value)
    recall_value = recall_score(df.label, df.predictionLabel)
    print("recall:", recall_value)

    fpr, tpr, thresholds = roc_curve(df.label, df.prediction)
    AUROC = auc(fpr, tpr)
    print("AUROC:", AUROC)

    precision, recall, thresholds = precision_recall_curve(df.label, df.prediction)
    AUPR = auc(recall, precision)
    print("AUPR:", AUPR)
    w=open('results/survyEval.tsv','a')
    w.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(model, species, precision_value, recall_value, AUROC, AUPR))

#Evaulation('results/ecoli_t5_MultiCNN2D_balanced_noValidation_3.tsv')
#Evaulation('results/human_t5_MultiCNN2D_balanced_noValidation_3_t5_MultiCNN1D_balanced_notValidation.tsv')
#Evaulation('../D-SCRIPT/predictions/Complete_ecoli_test.tsv')

#model_species_selection(species,models,'predictions')

def extendedEvaluationEPR(trueLable, prediction, predictedLabel):
    """
    Evaluate the performance of a binary classification model and print various metrics.

    Parameters:
    - trueLabel (array-like): True labels of the binary classification.
    - prediction (array-like): Predicted probability scores for the positive class.
    - predictedLabel (array-like): Predicted binary labels.

    Returns:
    Tuple containing evaluation metrics:
    - accuracy (float): Accuracy of the model.
    - sensitivity (float): Sensitivity (True Positive Rate/Recall) of the model.
    - specificity (float): Specificity (True Negative Rate) of the model.
    - precision (float): Precision of the model.
    - mcc (float): Matthews correlation coefficient.
    - f1 (float): F1 score of the model.
    - AUROC (float): Area Under the Receiver Operating Characteristic curve.
    - AUPR (float): Area Under the Precision-Recall curve.

    This function computes various classification metrics, prints them, and returns them as a tuple.
    It includes Sensitivity, Specificity, Precision, Matthews correlation coefficient, F1 score, 
    Accuracy, Area Under the ROC curve (AUROC), and Area Under the Precision-Recall curve (AUPR).
    """
    sorted_lst = sorted(prediction, reverse=True)

    # Calculate the index for the top 10%
    top_ten_percent_index = sorted_lst[int(len(sorted_lst) * sum(trueLable)/len(trueLable))]

    # Set the top 10% to 1 and the rest to 0
    result_lst = [1 if i > top_ten_percent_index else 0 for i in prediction]
    #print(sum(trueLable), sum(result_lst), len(trueLable), sum(trueLable)/len(trueLable))
    tn, fp, fn, tp = confusion_matrix(trueLable, result_lst).ravel()
    #print(tn, fp, fn, tp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    recall_val = tp / (tp + fn)
    precision_val = tp / (tp + fp)
    mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val)
    accuracy = (tp+tn)/(tp+tn+fp+fn)

    print("sensitivity: ", sensitivity)
    print("Specificity: ", specificity)
    #print("Recall: ", recall_val)
    print("Precision: ", precision_val)
    print("MCC: ", mcc)
    print("F1: ", f1)
    
    fpr, tpr, thresholds = roc_curve(trueLable, prediction)
    AUROC = auc(fpr, tpr)
    print("AUROC:", AUROC)

    precision, recall, thresholds = precision_recall_curve(trueLable, prediction)
    AUPR = auc(recall, precision)
    print("AUPR:", AUPR)

    return accuracy, sensitivity, specificity, precision_val, mcc, f1, AUROC, AUPR

def survey_extendedEvaluationEPR(trueLable, prediction, predictedLabel):
    """
    Evaluate the performance of a binary classification model and print various metrics.

    Parameters:
    - trueLabel (array-like): True labels of the binary classification.
    - prediction (array-like): Predicted probability scores for the positive class.
    - predictedLabel (array-like): Predicted binary labels.

    Returns:
    Tuple containing evaluation metrics:
    - accuracy (float): Accuracy of the model.
    - sensitivity (float): Sensitivity (True Positive Rate/Recall) of the model.
    - specificity (float): Specificity (True Negative Rate) of the model.
    - precision (float): Precision of the model.
    - mcc (float): Matthews correlation coefficient.
    - f1 (float): F1 score of the model.
    - AUROC (float): Area Under the Receiver Operating Characteristic curve.
    - AUPR (float): Area Under the Precision-Recall curve.

    This function computes various classification metrics, prints them, and returns them as a tuple.
    It includes Sensitivity, Specificity, Precision, Matthews correlation coefficient, F1 score, 
    Accuracy, Area Under the ROC curve (AUROC), and Area Under the Precision-Recall curve (AUPR).
    """
    sorted_lst = sorted(prediction, reverse=True)

    # Calculate the index for the top 10%
    top_ten_percent_index = sorted_lst[int(len(sorted_lst) * sum(trueLable)/len(trueLable))]

    # Set the top 10% to 1 and the rest to 0
    result_lst = [1 if i > top_ten_percent_index else 0 for i in prediction]
    #print(sum(trueLable), sum(result_lst), len(trueLable), sum(trueLable)/len(trueLable))
    tn, fp, fn, tp = confusion_matrix(trueLable, result_lst).ravel()
    #print(tn, fp, fn, tp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    recall_val = tp / (tp + fn)
    precision_val = tp / (tp + fp)
    mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val)
    accuracy = (tp+tn)/(tp+tn+fp+fn)

    print("sensitivity: ", sensitivity)
    print("Specificity: ", specificity)
    #print("Recall: ", recall_val)
    print("Precision: ", precision_val)
    print("MCC: ", mcc)
    print("F1: ", f1)
    
    fpr, tpr, thresholds = roc_curve(trueLable, prediction)
    AUROC = auc(fpr, tpr)
    print("AUROC:", AUROC)

    precision, recall, thresholds = precision_recall_curve(trueLable, prediction)
    AUPR = auc(recall, precision)
    print("AUPR:", AUPR)

    return accuracy, sensitivity, specificity, precision_val, mcc, f1, AUROC, AUPR


def model_species_selectionEPR(species, models, baseAddress):
    """
    Evaluate and compare different models for each species and save the results to a file.

    Parameters:
    - species (list): List of species names.
    - models (list): List of model names.
    - baseAddress (str): The base directory path where model results are stored.

    Returns:
    None

    This function evaluates and compares different models for each species based on their
    performance metrics such as accuracy, sensitivity, specificity, precision, MCC, F1 score,
    AUROC, and AUPR. The results are appended to an output file in tab-separated format.

    The file 'results/extendedEval.tsv' is created (or appended to) with a header line and
    subsequent lines containing the evaluation metrics for each combination of model and species.
    """
    w=open('results/extendedEvalERP.tsv','a')
    w.write("model\tkind\taccuracy\tsensitivity\tspecificity\tprecision\tmcc\tf1\tAUROC\tAUPR\n")
    for kind in species:
        for model in models:        
            df = pd.read_csv("{}/{}/{}.tsv".format(baseAddress,kind,model), sep='\t')
            accuracy, sensitivity, specificity, precision_val, mcc, f1, AUROC, AUPR = extendedEvaluationEPR(df.label, df.prediction ,df.predictionLabel)
            print(sensitivity, precision_val)
            w.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(model, kind, accuracy, 
                                                              sensitivity, specificity, precision_val, 
                                                              mcc, f1, AUROC, AUPR))
        w.write('\n')

def survey_model_species_selectionEPR(species, models):
    """
    Evaluate and compare different models for each species and save the results to a file.

    Parameters:
    - species (list): List of species names.
    - models (list): List of model names.
    - baseAddress (str): The base directory path where model results are stored.

    Returns:
    None

    This function evaluates and compares different models for each species based on their
    performance metrics such as accuracy, sensitivity, specificity, precision, MCC, F1 score,
    AUROC, and AUPR. The results are appended to an output file in tab-separated format.

    The file 'results/extendedEval.tsv' is created (or appended to) with a header line and
    subsequent lines containing the evaluation metrics for each combination of model and species.
    """
    w=open('results/surveyExtendedEvalERP.tsv','a')
    w.write("kind\tmodel\taccuracy\tsensitivity\tspecificity\tprecision\tmcc\tf1\tAUROC\tAUPR\n")
    for kind in species:
        for model in models:        
            df = pd.read_csv("results/survey_{}_{}_MLP.tsv".format(kind,model), sep='\t')
            accuracy, sensitivity, specificity, precision_val, mcc, f1, AUROC, AUPR = extendedEvaluationEPR(df.label, df.prediction ,df.predictionLabel)
            print(sensitivity, precision_val)
            w.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(kind, model, accuracy, 
                                                              sensitivity, specificity, precision_val, 
                                                              mcc, f1, AUROC, AUPR))
        w.write('\n')


def extendedEvaluationVthr(trueLable, prediction, thr):
    """
    Evaluate the performance of a binary classification model and print various metrics.

    Parameters:
    - trueLabel (array-like): True labels of the binary classification.
    - prediction (array-like): Predicted probability scores for the positive class.
    - predictedLabel (array-like): Predicted binary labels.

    Returns:
    Tuple containing evaluation metrics:
    - accuracy (float): Accuracy of the model.
    - sensitivity (float): Sensitivity (True Positive Rate/Recall) of the model.
    - specificity (float): Specificity (True Negative Rate) of the model.
    - precision (float): Precision of the model.
    - mcc (float): Matthews correlation coefficient.
    - f1 (float): F1 score of the model.
    - AUROC (float): Area Under the Receiver Operating Characteristic curve.
    - AUPR (float): Area Under the Precision-Recall curve.

    This function computes various classification metrics, prints them, and returns them as a tuple.
    It includes Sensitivity, Specificity, Precision, Matthews correlation coefficient, F1 score, 
    Accuracy, Area Under the ROC curve (AUROC), and Area Under the Precision-Recall curve (AUPR).
    """
    
    # Calculate the index for the top 10%
    predlabel = [1 if i > thr else 0 for i in prediction]
    #print(predlabel)
    #print(sum(trueLable), sum(result_lst), len(trueLable), sum(trueLable)/len(trueLable))
    tn, fp, fn, tp = confusion_matrix(trueLable, predlabel).ravel()
    #print(tn, fp, fn, tp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    recall_val = tp / (tp + fn)
    precision_val = tp / (tp + fp)
    mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val)
    accuracy = (tp+tn)/(tp+tn+fp+fn)

    print("sensitivity: ", sensitivity)
    print("Specificity: ", specificity)
    #print("Recall: ", recall_val)
    print("Precision: ", precision_val)
    print("MCC: ", mcc)
    print("F1: ", f1)
    
    fpr, tpr, thresholds = roc_curve(trueLable, prediction)
    AUROC = auc(fpr, tpr)
    print("AUROC:", AUROC)

    precision, recall, thresholds = precision_recall_curve(trueLable, prediction)
    AUPR = auc(recall, precision)
    print("AUPR:", AUPR)

    return accuracy, sensitivity, specificity, precision_val, mcc, f1, AUROC, AUPR

def model_species_selectionVthr(species, models, baseAddress):
    w=open('results/extendedEval.tsv','a')
    w.write("model\tkind\tthr\tprecision\trecall\tAUROC\tAUPR\n")
    
    for kind in species:
        maxthrf1 = 0
        for model in models:
            for thr in range(50,100,1):        
                df = pd.read_csv("{}/{}/{}.tsv".format(baseAddress,kind,model), sep='\t')
                accuracy, sensitivity, specificity, precision_val, mcc, f1, AUROC, AUPR = extendedEvaluationVthr(df.label, df.prediction ,thr/100)
                #print(sensitivity, precision_val)
                w.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(model, kind, thr/100, precision_val, sensitivity,  AUROC, AUPR, f1))
        w.write('\n')




species = ['human']
models = ['Ensemble+puzzler']

species = ['ecoli','yeast','fly','mouse','worm', 'human']
models = ['t5','bepler', 'xlnet', 'plus', 'bert', 'albert' ] 
#models = ['CNN1D', 'CNN2D', 'Ensemble','CNN1D+puzzler', 'CNN2D+puzzler', 
#          'Ensemble+puzzler', 'D-SCRIPT']
   #model_species_selectionEPR(species,models,'predictions')    
#model_species_selectionVthr(species,models,'predictions')    

if __name__ == "__main__":
    survey_model_species_selectionEPR(species,models)    
