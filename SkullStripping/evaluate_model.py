import keras
import numpy as np
import argparse
from Unet import Predictor3DUnet
from CNN import Predictor3DCNN
import helper
import pickle

# TODO: add save predictions?
def evaluate(predicting_arc, save_name, data, labels, evaluating_with_slurm, d):
    dcs_list = []
    sen_list = []
    spe_list = []

    score_file = helper.open_score_file(save_name, evaluating_with_slurm)
    score_file.write("name\tdcs\tsen\tspe\n")
    
    for i in range(0, len(data)):
        print("Evaluating", d[i])
        pred = predicting_arc.predict_data(predicting_arc.model, data[i], predicting_arc.input_size[:3])
        pred = (pred > 0.5).astype('int8')
        dsc, sen, spe = compute_scores(pred, labels[i])
        print("Dice score for " + d[i] + ": " + str(dsc))
        score_file.write(d[i] + "\t" + str(dsc) + "\t" + str(sen) + "\t" + str(spe) + "\n")
        
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
    print(label.shape)
    print(pred.shape)
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
    
    if((2 * TP + FP + FN) == 0):
        dice_coefficient = 1.0
    else:
        dice_coefficient = (2 * TP) / (2 * TP + FP + FN)
    if((TP + FN) == 0):
        sensitivity = 1.0
    else:
        sensitivity = TP / (TP + FN)
    if((TN + FP) == 0):
        specificity = 1.0
    else:
        specificity = TN / (TN + FP)
     
    return dice_coefficient, sensitivity, specificity

def main():
    parser = argparse.ArgumentParser(description='Module for training a model or predicting using an existing model')
    parser.add_argument('--arc', dest='arc', required=True, type=str, help='Specify which arcitecture')
    parser.add_argument('--savename', dest='save_name', required=True, type=str, help='Path to the corresponding labels')
    parser.add_argument('--data', dest='data', required=True, type=str, nargs='+', help='Path to the data')
    parser.add_argument('--labels', dest='labels', required=True, type=str, nargs='+', help='The save name of the model')
    parser.add_argument("--gpus", dest='gpus', required=True, type=int, default=1, help="# of GPUs to use for training")
    
    parser.add_argument("--use_testing_data", dest='use_testing_data', required=True, type=bool, default=False, help="# of GPUs to use for training")
    parser.add_argument("--evaluating_with_slurm", dest='evaluating_with_slurm', required=False, type=bool, default=False, help="# of GPUs to use for training")
    args = parser.parse_args()
    
    d = np.asarray(helper.load_files(args.data))
    l = np.asarray(helper.load_files(args.labels))
    
    if(args.evaluating_with_slurm):
        data, labels = helper.load_data_and_labels(d, l)
    else:
        data, labels = helper.patchCreator(d, l, normalize=True)
    
    if(args.arc == 'unet'):
        unet = Predictor3DUnet.Predictor3DUnet(args.save_name, (64, 64, 64, 1), args.gpus, evaluating_with_slurm=args.evaluating_with_slurm)
        if(args.use_testing_data):
            testing_indices = helper.load_indices(args.save_name, "testing_indices", evaluating_with_slurm=args.evaluating_with_slurm)
            evaluate(unet, args.save_name, data[testing_indices], labels[testing_indices], args.evaluating_with_slurm, d[testing_indices])
        else:
            evaluate(unet, args.save_name, data, labels, args.evaluating_with_slurm, d)

    elif(args.arc == 'cnn'):
        # Apply cc filtering should maybe be here.
        cnn = Predictor3DCNN.Predictor3DCNN(args.save_name, args.gpus, evaluating_with_slurm=args.evaluating_with_slurm)
        if(args.use_testing_data):
            testing_indices = helper.load_indices(args.save_name, "testing_indices", evaluating_with_slurm=args.evaluating_with_slurm)
            evaluate(cnn, args.save_name, data[testing_indices], labels[testing_indices], args.evaluating_with_slurm, d[testing_indices])
        else:
            evaluate(cnn, args.save_name, data, labels, args.evaluating_with_slurm, d)
main()