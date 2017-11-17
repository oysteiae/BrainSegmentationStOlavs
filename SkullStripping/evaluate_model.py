import keras
import numpy as np
from SkullStripping import patchCreator
from SkullStripping import compute_scores
from SkullStripping import load_files
from SkullStripping import run_on_block
from SkullStripping import buildCNN
from SkullStripping import remove_small_conneceted_components

def predict(data, model,apply_cc_filtering=True, using_sparse_categorical_crossentropy=False):
    sav = run_on_block(model, data)
    
    # TODO understand what this does
    if(apply_cc_filtering):
        predicted = remove_small_conneceted_components(sav)
        predicted = 1 - remove_small_conneceted_components(1 - sav)

    # Adding extra chanel so that it has equal shape as the input data.
    predicted = np.expand_dims(predicted, axis=4)

    if(using_sparse_categorical_crossentropy):
        sav = (predicted <= 0.5).astype('int8')
    else:
        sav = (predicted > 0.5).astype('int8')

    return sav


def main():
    kdr1 = [0, 1, 4, 6, 7, 8, 9, 10, 14, 16, 19, 23, 24,  25,  28,  31,  34,  35, 38,  39,  41,  42,  44,  47,  48,  53,  55,  56,  57,  60,  61,  63,  64,  65, 67,  68, 69,  72,  73,  75,  80,  83,  85,  88,  89,  90,  92,  94,  95,  96,  97, 103, 104, 105,108, 110, 112, 114]
    kdr2 = [2, 3, 5, 11,  12,  13,  15,  17,  18,  20,  21,  22,  26,  27,  29,  30,  32,  33, 36,  37,  40,  43,  45,  46,  49,  50,  51,  52,  54,  58,  59,  62,  66,  70,  71,  74, 76,  77,  78,  79,  81,  82,  84,  86,  87,  91,  93,  98,  99, 100, 101, 102, 106, 107, 109, 111, 113, 115]
    print(sum(kdr1))
    d = load_files(["C:\\Users\\oyste\\OneDrive\\MRI_SCANS\\data"])
    #don't really need this but patchcreator needs to be rewritten to remove it
    l = load_files(["C:\\Users\\oyste\\OneDrive\\MRI_SCANS\\labels"])
    data, labels = patchCreator(d, l)
    
    print("Predicting with model one")
    test_data_model_one = data[kdr2]
    test_data_model_one_labels = labels[kdr2]
    dcs_model_one_list = []
    sen_model_one_list = []
    spe_model_one_list = []
    cnn_input_size = (84, 84, 84, 1)


    score_file = open("scores.tsv", 'w')
    score_file.write("dcs\tsen\tspe\n")
    
    model_one = buildCNN(cnn_input_size)
    model_one.load_weights("kdr2" + ".h5")
    for i in range(0, len(test_data_model_one)):
        pred = predict(test_data_model_one[i], model_one)
        dsc, sen, spe = compute_scores(pred, test_data_model_one_labels[i])
        print("Dice score for " + str(i) + str(dsc))
        score_file.write(str(dsc) + "\t" + str(sen) + "\t" + str(spe) + "\n")
        
        dcs_model_one_list.append(dsc)
        sen_model_one_list.append(sen)
        spe_model_one_list.append(spe)

    test_data_model_two = data[kdr1]
    test_data_model_two_labels = labels[kdr1]
    dcs_model_two_list = []
    sen_model_two_list = []
    spe_model_two_list = []

    model_two = buildCNN(cnn_input_size)
    model_two.load_weights("kdr1" + ".h5")
    for i in range(0, len(test_data_model_two)):
        pred = predict(test_data_model_two[i], model_two)
        dsc, sen, spe = compute_scores(pred, test_data_model_two_labels[i])
        print("Dice score for " + str(i) + str(dsc))
        score_file.write(str(dsc) + "\t" + str(sen) + "\t" + str(spe) + "\n")
        
        dcs_model_two_list.append(dsc)
        sen_model_two_list.append(sen)
        spe_model_two_list.append(spe)
    
    score_file.close()

    average_dice = (sum(dcs_model_one_list) + sum(dcs_model_one_list)) / len(data)
    average_sen = (sum(sen_model_one_list) + sum(sen_model_two_list)) / len(data)
    average_spe = (sum(sen_model_one_list) + sum(sen_model_two_list)) / len(data)

    print("Average dice score:", average_dice)
    print("Average sensitivity:", average_sen)
    print("Average specificity:", average_spe)

main()