import keras
import numpy as np
import argparse
import Unet
import CNN
import helper

def evaluate(model, predicting_function, save_name):
    dcs_list = []
    sen_list = []
    spe_list = []

    score_file = open("scores_" + save_name + ".tsv", 'w')
    score_file.write("dcs\tsen\tspe\n")
    
    for i in range(0, len(data)):
        pred = predicting_function(data[i], model_one)
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

def main():
    parser = argparse.ArgumentParser(description='Module for training a model or predicting using an existing model')
    parser.add_argument('--arc', dest='arc', required=True, type=str, help='Specify which arcitecture')
    parser.add_argument('--savename', dest='save_name', required=True, type=str, help='Path to the corresponding labels')
    parser.add_argument('--data', dest='data', required=True, type=str, nargs='+', help='Path to the data')
    parser.add_argument('--labels', dest='labels', required=True, type=str, nargs='+', help='The save name of the model')
    parser.add_argument("--gpus", dest='gpus', required=True, type=int, default=1, help="# of GPUs to use for training")
    args = parser.parse_args()
    
    d = helper.load_files(args.data)
    l = helper.load_files(args.labels)
    data, labels = helper.patchCreator(d, l, normalize=True)
    
    if(args.arc == 'unet'):
        model = Unet.Build3DUnet.build_3DUnet((64, 64, 64, 1), args.gpus)
        model.load_weights(args.save_name + ".h5")
        predicting_function = Unet.Predictor3DUnet.predict_data_from_patches()
        evaluate(model, predicting_function, args.save_name)
    elif(args.arc == 'cnn'):
        # Apply cc filtering should maybe be here.
        model = CNN.Build3DCNN.build_3DCNN((84, 84, 84, 1), args.gpus)
        model.load_weights(args.save_name + ".h5")
        predicting_function = CNN.Predictor3DCNN.run_on_block()
        evaluate(model, predicting_function, args.save_name)
main()