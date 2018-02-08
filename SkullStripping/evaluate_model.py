import keras
import numpy as np
from helper import patchCreator
from SkullStripping import load_files
from SkullStripping import run_on_block
from SkullStripping import build_CNN
from SkullStripping import remove_small_connected_components

# This whole class should be rewritten.
def predict(data, model,apply_cc_filtering=True, using_sparse_categorical_crossentropy=False):
    sav = run_on_block(model, data)
    
    # TODO understand what this does
    if(apply_cc_filtering):
        predicted = remove_small_connected_components(sav)
        predicted = 1 - remove_small_connected_components(1 - sav)

    # Adding extra chanel so that it has equal shape as the input data.
    predicted = np.expand_dims(predicted, axis=4)

    if(using_sparse_categorical_crossentropy):
        sav = (predicted <= 0.5).astype('int8')
    else:
        sav = (predicted > 0.5).astype('int8')

    return sav

def evaluate_cross_entropy():
    test_data_model_one_indexes = [76,77,78,79,81,82,84,86,87,91,93,98,99,100,101,102,106,107,109,111,113,115] #[2,3,5,11,12,13,15,17,18,20,21,22,26,27,29,30,32,33,36,37,40,43,45,46,49,50,51,52,54,58,59,62,66,70,71,74]#,76,77,78,79,81,82,84,86,87,91,93,98,99,100,101,102,106,107,109,111,113,115]
    test_data_model_two_indexes = [80,83,85,88,89,90,92,94,95,96,97,103,104,105,108,110,112,114]#[0,1,4,6,7,8,9,10,14,16,19,23,24,25,28,31,34,35,38,39,41,42,44,47,48,53,55,56,57,60,61,63,64,65,67,68,69,72,73,75]#,80,83,85,88,89,90,92,94,95,96,97,103,104,105,108,110,112,114]
    
    d = load_files(["C:\\Users\\oyste\\OneDrive\\MRI_SCANS\\data"])
    #don't really need this but patchcreator needs to be rewritten to remove it
    l = load_files(["C:\\Users\\oyste\\OneDrive\\MRI_SCANS\\labels"])
    data, labels = patchCreator(d, l)
    
    print("Predicting with model one")
    test_data_model_one = data[test_data_model_one_indexes]
    test_data_model_one_labels = labels[test_data_model_one_indexes]
    dcs_list = []
    sen_list = []
    spe_list = []
    cnn_input_size = (84, 84, 84, 1)

    score_file = open("scores_crossvalidationLBPA40.tsv", 'w')
    score_file.write("dcs\tsen\tspe\n")
    
    model_one = build_CNN(cnn_input_size)
    model_one.load_weights("both_datasets_test1" + ".h5")
    for i in range(0, len(test_data_model_one)):
        pred = predict(test_data_model_one[i], model_one)
        dsc, sen, spe = compute_scores(pred, test_data_model_one_labels[i])
        print("Dice score for " + str(i) + str(dsc))
        score_file.write(str(dsc) + "\t" + str(sen) + "\t" + str(spe) + "\n")
        
        dcs_list.append(dsc)
        sen_list.append(sen)
        spe_list.append(spe)

    test_data_model_two = data[test_data_model_two_indexes]
    test_data_model_two_labels = labels[test_data_model_two_indexes]

    model_two = build_CNN(cnn_input_size)
    model_two.load_weights("both_datasets_test2" + ".h5")
    for i in range(0, len(test_data_model_two)):
        pred = predict(test_data_model_two[i], model_two)
        dsc, sen, spe = compute_scores(pred, test_data_model_two_labels[i])
        print("Dice score for " + str(i) + ": " + str(dsc))
        score_file.write(str(dsc) + "\t" + str(sen) + "\t" + str(spe) + "\n")
        
        dcs_list.append(dsc)
        sen_list.append(sen)
        spe_list.append(spe)
    
    score_file.close()

    average_dice = sum(dcs_list) / len(dcs_list)
    average_sen = sum(sen_list) / len(sen_list)
    average_spe = sum(spe_list) / len(spe_list)

    print("Average dice score:", average_dice)
    print("Average sensitivity:", average_sen)
    print("Average specificity:", average_spe)

def evaluate():
    d = load_files(["C:\\Users\\oyste\\OneDrive\\MRI_SCANS\\LBPA40_data"])
    #don't really need this but patchcreator needs to be rewritten to remove it
    l = load_files(["C:\\Users\\oyste\\OneDrive\\MRI_SCANS\\LBPA40_labels"])
    data, labels = patchCreator(d, l)
    
    dcs_list = []
    sen_list = []
    spe_list = []
    cnn_input_size = (84, 84, 84, 1)

    score_file = open("scores_Experiment3_oasis_model_lbpa40data.tsv", 'w')
    score_file.write("dcs\tsen\tspe\n")
    
    model_one = build_CNN(cnn_input_size)
    model_one.load_weights("Experiment1_onlyoasis" + ".h5")
    for i in range(0, len(data)):
        pred = predict(data[i], model_one)
        dsc, sen, spe = compute_scores(pred, labels[i])
        print("Dice score for " + str(i) + ": " + str(dsc))
        score_file.write(str(dsc) + "\t" + str(sen) + "\t" + str(spe) + "\n")
        
        dcs_list.append(dsc)
        sen_list.append(sen)
        spe_list.append(spe)

    average_dice = sum(dcs_list) / len(data)
    average_sen = sum(sen_list) / len(data)
    average_spe = sum(spe_list) / len(data)

    score_file.close()

    print("Average dice score:", average_dice)
    print("Average sensitivity:", average_sen)
    print("Average specificity:", average_spe)

# TODO: I think this can be sped up by calculating it differntly
def compute_scores(pred, label):
    assert pred.shape == label.shape, "Shape mismatch between prediction and label when calculating scores"
    
    shape = pred.shape
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(0, shape[0]):
        if(i % 25 == 0):
            print("Comleted", float(i)/float(shape[0]) * 100, "%")
        for j in range(0, shape[1]):
            for k in range(0, shape[2]):
                if(pred[i][j][k] == 1 and label[i][j][k] >= 1):
                    TP += 1
                elif(pred[i][j][k] == 1 and label[i][j][k] == 0):
                    FP += 1
                elif(pred[i][j][k] == 0 and label[i][j][k] >= 1):
                    FN += 1  
                elif(pred[i][j][k] == 0 and label[i][j][k] == 0):
                    TN += 1

    dice_coefficient = (2 * TP) / (2 * TP + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
     
    return dice_coefficient, sensitivity, specificity