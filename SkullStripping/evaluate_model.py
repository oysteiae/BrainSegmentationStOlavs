import keras
import numpy as np
import argparse
from Unet import Predictor3DUnet
from CNN import Predictor3DCNN
import helper
import pickle

# TODO: add save predictions?
def evaluate(predicting_arc, save_name, data, labels, evaluating_with_slurm, d, part_to_test_on, save_predictions):
    dcs_list = []
    sen_list = []
    spe_list = []

    score_file = helper.open_score_file(save_name, evaluating_with_slurm, part_to_test_on)
    score_file.write("name\tdcs\tsen\tspe\n")
    
    for i in range(0, len(data)):
        print("Evaluating", d[i])
        pred = predicting_arc.predict_data(predicting_arc.model, data[i], predicting_arc.input_size[:3])
        if(save_predictions):
            helper.save_prediction(ntpath.basename(d[i]).split('.')[0], pred, save_name + "_pred_", original_file=d[i])

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
    
    parser.add_argument("--use_testing_data", dest='use_testing_data', required=False, type=bool, default=False, help="# of GPUs to use for training")
    parser.add_argument("--evaluating_with_slurm", dest='evaluating_with_slurm', required=False, type=bool, default=False, help="# of GPUs to use for training")
    
    parser.add_argument('--patch_size', dest='patch_size', required=False, type=int, nargs='+', help='Size of patch used for input, default to (59, 59, 59, 1) for CNN and (64, 64, 64, 1) for the U-Net')
    parser.add_argument('--loss_function', dest='loss_function', required=False, type=str, help='The loss function to use, defaults to kld')
    parser.add_argument('--part_to_test_on', dest='part_to_test_on', required=False, type=str, help='Test on the training, training or testing part of the data if use training data is True')
    parser.add_argument('--location_previous_training_and_validation_indices', dest='location_previous_training_and_validation_indices', required=False, type=str, help='Which experiment to load previous training and validation indices from')
    parser.add_argument('--save_predictions', dest='save_predictions', required=False, type=bool, default=False, help='True if you the predictions should be saved during training.')


    args = parser.parse_args()
    
    d = np.asarray(helper.load_files(args.data))
    l = np.asarray(helper.load_files(args.labels))
    
    if(args.part_to_test_on is None and args.use_testing_data):
        part_to_test_on = 'testing_indices'
    elif(args.use_testing_data):
        part_to_test_on = args.part_to_test_on + "_indices"
    else:
        part_to_test_on = None
    
    if(args.location_previous_training_and_validation_indices is None):
        location_previous_training_and_validation_indices = args.save_name
    else:
        location_previous_training_and_validation_indices = args.location_previous_training_and_validation_indices

    if(args.use_testing_data):
        testing_indices = helper.load_indices(location_previous_training_and_validation_indices, part_to_test_on, evaluating_with_slurm=args.evaluating_with_slurm)
        data, labels = helper.patchCreator(d[testing_indices], l[testing_indices], normalize=True)
        d = d[testing_indices]
    else:
        data, labels = helper.patchCreator(d, l, normalize=True)
    
    if(args.arc == 'unet'):
        if(args.patch_size == None):
            patch_size = (64, 64, 64, 1)
        else:
            patch_size = tuple(args.patch_size)
        
        unet = Predictor3DUnet.Predictor3DUnet(args.save_name, patch_size, args.gpus, evaluating_with_slurm=args.evaluating_with_slurm, loss_function=args.loss_function)
        evaluate(unet, args.save_name, data, labels, args.evaluating_with_slurm, d, part_to_test_on, args.save_predictions)

    elif(args.arc == 'cnn'):
        if(args.patch_size == None):
            patch_size = (84, 84, 84, 1)
        else:
            patch_size = tuple(args.patch_size)

        # Apply cc filtering should maybe be here.
        cnn = Predictor3DCNN.Predictor3DCNN(args.save_name, args.gpus, evaluating_with_slurm=args.evaluating_with_slurm, input_size=patch_size, loss_function=args.loss_function)
        evaluate(cnn, args.save_name, data, labels, args.evaluating_with_slurm, d, part_to_test_on, args.save_predictions)
main()