# Train script for replicating TractSeg original paper
import numpy as np
import os
import torch
import time
import importlib

from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

from argparse import ArgumentParser

def get_dice(result, label):
    dice = 0
    num_subjects = len(result)
    count = 0
    for subject_num in range(num_subjects):
        for dim in range(72):
            a, b = result[subject_num,dim,:,:], label[subject_num,dim,:,:]
            TP = torch.sum(((a+b)==2).float())
            FP = torch.sum(((a+(1-b))==2).float())
            FN = torch.sum(((b+(1-a))==2).float())

            if TP + FP + FN == 0:
                tract_dice = TP*0 + 1 # so compatible types
            else:
                tract_dice = 2*TP/(2*TP+FP+FN)
            dice += tract_dice
            count += 1

    return dice.item()/count

def get_random_valid_fold(EVAL_FOLD):
    EVAL_FOLD = EVAL_FOLD - 1 # adjust to range [0,4]

    folds = [1,2,3,4,5]
    non_eval_folds = np.array(folds[:EVAL_FOLD] + folds[EVAL_FOLD+1:])
    
    np.random.shuffle(non_eval_folds)
    return non_eval_folds[0]

# Random number seeding
np.random.seed(66)
torch.manual_seed(66)

RESULTS_FOLDER = './results/'
FOLDS_PATH = 'G:/merged_dataset_preprocessed_slices_numpy_efficient/folds'

parser = ArgumentParser(description="Arguments for model training.")
parser.add_argument("-j", "--job_mode", help="Presence of this flag enables job mode for cloud jobs (turns off visualisation, etc.).", action='store_true')
parser.add_argument("-r", "--results_name", help="Name of folder to save results to. Folder will be created. If --fn_to_load is specified, this will be the folder containing the checkpoint to load.", default='dump', type=str)
parser.add_argument("-ef", "--eval_fold", help="Fold used for evaluation. All data from this fold will be excluded from this training procedure entirely (it is a hold-out set).", default=3, type=int)
parser.add_argument("-b", "--batch_size", help="Batch size.", default=10, type=int)
parser.add_argument("-lr", "--learning_rate", help="Learning rate.", default=1e-3, type=float)
parser.add_argument("-e", "--epochs", help="Number of epochs to train for.", default=250, type=int)
parser.add_argument("-aug", "--augmentation", help="Presence of this flag turns on data augmentation.", action='store_true')
parser.add_argument("-m", "--module_name", help="Specify the name of the file containing the model.", default='genotype', type=str)
parser.add_argument("-l", "--fn_to_load", help="Specify the name of the file in results/model_name to load and resume training. Include extensions, e.g. 'epoch_10.pth'. Exclude this argument if you dont want to load anything.", default='', type=str)

args = parser.parse_args()
job_mode = args.job_mode
model_name = args.results_name
EVAL_FOLD = args.eval_fold
BATCH_SIZE = args.batch_size
LR = args.learning_rate
EPOCHS = args.epochs
data_augmentation = args.augmentation
module_name = args.module_name
VALID_FOLD = get_random_valid_fold(EVAL_FOLD)
fn_to_load = args.fn_to_load
if fn_to_load != '':
    to_load = True

    model_fn = RESULTS_FOLDER + model_name + '/' + fn_to_load
    m = torch.load(model_fn)
    EVAL_FOLD = int(m['eval_fold'])
    BATCH_SIZE = int(m['batch_size'])
    LR = float(m['lr'])
    data_augmentation = m['data_augmentation']
    module_name = m['module_name']
    VALID_FOLD = int(m['valid_fold'])
else:
    to_load = False



if not job_mode:
    from resources.vis import VisdomLinePlotter
    import cv2

# MODEL
mod = importlib.import_module('models.' + module_name)
from models.TRAIN_DATA_LOADER import CustomDataset

if not job_mode:
    print('Initialising visualisation...')
    plotter = VisdomLinePlotter(env_name='TractSeg Replication')
    dice_plotter = VisdomLinePlotter(env_name='TractSeg Replication')

"""
###################
# DATASET LOADING #
###################
"""

"""
### Create the dataset ###
Two separate datasets are created for the training and validation sets so that data augmentation
is turned off for validation set.
"""
#means = np.array([1,1,1,1,1,1,1,1,1])
#sdevs = np.array([0,0,0,0,0,0,0,0,0])
train_dataset = CustomDataset(FOLDS_PATH, eval_fold=EVAL_FOLD, valid_fold=VALID_FOLD, do_augment=data_augmentation)
dataset_means, dataset_sdevs = train_dataset.get_std_params()
valid_dataset = CustomDataset(FOLDS_PATH, eval_fold=EVAL_FOLD, valid_fold=VALID_FOLD, is_validation=True, do_augment=False, means=dataset_means, sdevs=dataset_sdevs)

# Create the data samplers
train_indices = list(range(len(train_dataset)))
val_indices = list(range(len(valid_dataset)))
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

# Create the data loaders
trainloader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE, drop_last=True)
validloader = torch.utils.data.DataLoader(valid_dataset, sampler=valid_sampler, batch_size=BATCH_SIZE, drop_last=True)

print('Validation fold %d selected.' % (VALID_FOLD))
print(str(len(trainloader)*BATCH_SIZE) + ' training items (drop_last=True)')
print(str(len(validloader)*BATCH_SIZE) + ' validation items (drop_last=True)')

"""
#############################
# MODEL AND TRAINER LOADING #
#############################
"""
torch.cuda.empty_cache()
device = torch.device('cuda')
print("Using device: ", device)

if to_load == True:
    # Load the pre-trained model
    checkpoint = m
    model = mod.CustomModel()
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.Adamax(model.parameters(), lr=LR)
    optimizer.load_state_dict(checkpoint['optimizer'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=20)
    scheduler.load_state_dict(checkpoint['scheduler'])
else:
    # Create the model
    model = mod.CustomModel()

    # Create the results directory
    if model_name != "dump":
        while os.path.exists(RESULTS_FOLDER + model_name):
            model_name = input('Model already exists. Please enter another name:')
        os.mkdir(RESULTS_FOLDER + model_name) 
    else:
        if not os.path.exists(RESULTS_FOLDER + 'dump'):
            os.mkdir(RESULTS_FOLDER + 'dump')
    optimizer = torch.optim.Adamax(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=20)

model.to(device)

# Print a model summary
if not job_mode:
    from torchsummary import summary
    summary(model, (9,144,144))
torch.cuda.empty_cache()

# Load history of losses and dice scores
start_epoch = 1
if to_load == True:
    start_epoch = int(m['training_epoch']) + 1

    # If using new loading scheme, load directly from save file
    if 'train_losses' in m:
        training_losses = m['train_losses']
        valid_losses = m['valid_losses']
        dice_scores = m['valid_dices']
    else:
        training_losses = list(np.load(RESULTS_FOLDER + model_name + '/training_losses.npy'))[:m['training_epoch']]
        valid_losses = list(np.load(RESULTS_FOLDER + model_name + '/valid_losses.npy'))[:m['training_epoch']]
        dice_scores = list(np.load(RESULTS_FOLDER + model_name + '/dice_scores.npy')[:m['training_epoch']])

    # Plot the loss history
    if not job_mode:
        for e in range(1,len(training_losses)+1):
            plotter.plot('loss', 'training', 'Results', e, training_losses[e-1])
            plotter.plot('loss', 'validation', 'Results', e, valid_losses[e-1])
            dice_plotter.plot('dice score', 'validation', 'Dice Score', e, dice_scores[e-1], 'batch number')
else:
    training_losses = []
    valid_losses = []
    dice_scores = []

best_valid_dice = 0
if len(dice_scores) != 0:
    best_valid_dice = max(dice_scores)

"""
############
# TRAINING #
############
"""
print("Training...")
if not job_mode:
    cv2.namedWindow('input_data', cv2.WINDOW_NORMAL)
    cv2.namedWindow('raw_output', cv2.WINDOW_NORMAL)
    cv2.namedWindow('sigmoid_output', cv2.WINDOW_NORMAL)
    cv2.namedWindow('thresh_output', cv2.WINDOW_NORMAL)
    cv2.namedWindow('ground_truth', cv2.WINDOW_NORMAL)

for epoch in range(start_epoch, EPOCHS+1):
    t0 = time.time()

    train_loss = 0.0
    num_train_slices = 0
    model.train()

    print('EPOCH ' + str(epoch) + '...')
    batch_num = 0
    for inputs, labels in trainloader:
        #if epoch == start_epoch:
        #    num_train_slices = 1
        #    batch_num = 1
        #    break
        batch_num += 1

        # Forward propagation
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model.forward(inputs)
        output_sigmoid = F.sigmoid(output).detach() if type(output) != type([]) else [F.sigmoid(out).detach() for out in output]
        
        # VISUALISATION
        if not job_mode:
            if type(output) == type([]): # only vis the last output if output is a list (e.g. for deep supervision)
                output_vis = output[-1]
                output_sigmoid_vis = output_sigmoid[-1]
            else:
                output_vis = output
                output_sigmoid_vis = output_sigmoid
            if batch_num % 2 == 0:
                vis_input_data = inputs.detach().permute(0,2,3,1).cpu().numpy()[0][:,:,:3]
                vis_raw_output = output_vis.detach().permute(0,2,3,1).cpu().numpy()[0][:,:,:3]
                vis_sigmoid_output = output_sigmoid_vis.detach().permute(0,2,3,1).cpu().numpy()[0][:,:,:3]

                vis_thresh_output = vis_sigmoid_output.copy()
                vis_thresh_output[vis_thresh_output > 0.5] = 255
                vis_thresh_output[vis_thresh_output < 255] = 0
                vis_ground_truth = labels.detach().permute(0,2,3,1).cpu().numpy()[0][:,:,:3] * 255 // 1

                # 2. Normalise outputs to range 0,255
                vis_input_data = (vis_input_data - np.min(vis_input_data)) / (np.max(vis_input_data) - np.min(vis_input_data)) * 255 // 1
                vis_raw_output = (vis_raw_output - np.min(vis_raw_output)) / (np.max(vis_raw_output) - np.min(vis_raw_output)) * 255 // 1
                vis_sigmoid_output = (vis_sigmoid_output - np.min(vis_sigmoid_output)) / (np.max(vis_sigmoid_output) - np.min(vis_sigmoid_output)) * 255 // 1

                cv2.imshow('input_data', np.uint8(vis_input_data))
                cv2.imshow('raw_output', np.uint8(vis_raw_output))
                cv2.imshow('sigmoid_output', np.uint8(vis_sigmoid_output))
                cv2.imshow('thresh_output', np.uint8(vis_thresh_output))
                cv2.imshow('ground_truth', np.uint8(vis_ground_truth))
                cv2.waitKey(1)

        # Compute loss and propagate backward
        loss = mod.CustomLoss(output, labels)
        loss_item = loss.detach().item()
        loss.backward()

        # Optimise weights
        optimizer.step()
     
        # Update loss record
        train_loss  += loss_item
        num_train_slices += inputs.size(0)

        torch.cuda.empty_cache()

    train_time = time.time() - t0
    print('##### TRAINING TIME #####')
    print('Per batch: %.3f seconds' % (train_time / batch_num))
    print('Per epoch: %.3f minutes' % (train_time / 60))
    print('Per %d epochs: %.3f hours' % (EPOCHS, train_time * EPOCHS / 60 / 60))
    print('#########################')

    # Validation
    valid_loss = 0.0
    num_valid_slices = 0
    dice_score = 0.0
    num_dice_scores = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model.forward(inputs)
            output_sigmoid = F.sigmoid(output).detach() if type(output) != type([]) else [F.sigmoid(out).detach() for out in output]

            loss = mod.CustomLoss(output, labels)
            loss_item = loss.detach().item()

            # Calculate dice score
            if type(output_sigmoid) == type([]):
                output_sigmoid = output_sigmoid[-1]
            output_thresh = (output_sigmoid>0.5).float()
            target_thresh = (labels>0.5).float()
            dice = get_dice(output_thresh, target_thresh)

            dice_score += dice
            num_dice_scores += 1

            valid_loss += loss_item
            num_valid_slices += inputs.size(0)

        torch.cuda.empty_cache()
            
    scheduler.step(valid_loss)

    print("Epoch %d/%d:\t%.5f\t%.5f" % (epoch, EPOCHS, train_loss/num_train_slices, valid_loss/num_valid_slices))

    # Save the training/valid losses to a file
    training_losses.append(train_loss/num_train_slices)
    valid_losses.append(valid_loss/num_valid_slices)
    dice_scores.append(dice_score/num_dice_scores)

    if not job_mode:
        plotter.plot('loss', 'training', 'Results', epoch, train_loss/num_train_slices)
        plotter.plot('loss', 'validation', 'Results', epoch, valid_loss/num_valid_slices)
        dice_plotter.plot('dice score', 'validation', 'Dice Score', epoch, dice_score/num_dice_scores, 'batch number')

    np.save(RESULTS_FOLDER + model_name + '/training_losses', np.array(training_losses))
    np.save(RESULTS_FOLDER + model_name + '/valid_losses', np.array(valid_losses))
    np.save(RESULTS_FOLDER + model_name + '/dice_scores', np.array(dice_scores))
    np.save(RESULTS_FOLDER + model_name + '/lr.npy', LR)

    save_object = {
        'model_state_dict':model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'training_epoch': epoch,
        'valid_loss': train_loss/num_train_slices,
        'train_loss': valid_loss/num_valid_slices,
        'valid_dice': dice_score/num_dice_scores,
        'model_name': model_name,
        'eval_fold': EVAL_FOLD,
        'valid_fold': VALID_FOLD,
        'batch_size': BATCH_SIZE,
        'lr': LR,
        'epochs': EPOCHS,
        'data_augmentation': data_augmentation,
        'module_name': module_name,
        'means': dataset_means,
        'sdevs': dataset_sdevs,
        'train_losses': training_losses,
        'valid_losses': valid_losses,
        'valid_dices': dice_scores,
    }

    if epoch % 10 == 0 or epoch <= 20:
        print('Saving intermediate model...')
        torch.save(save_object, RESULTS_FOLDER + model_name + '/epoch_' + str(epoch) + '.pth')

    # Always save the most recent epoch
    print('Saving current model...')
    torch.save(save_object, RESULTS_FOLDER + model_name + '/epoch_current.pth')

    # Save the current epoch as the best epoch if it has the best validation dice score
    if dice_score/num_dice_scores > best_valid_dice:
        torch.save(save_object, RESULTS_FOLDER + model_name + '/best_epoch.pth')
        best_valid_dice = dice_score/num_dice_scores

    total_epoch_time = time.time() - t0
    print('##### TOTAL TRAIN + VALIDATION TIME #####')
    print('Per epoch: %.1f mins' % (total_epoch_time / 60))
    print('Per %d epochs: %.1f hours' % (EPOCHS, total_epoch_time*EPOCHS/60/60))
    print('#########################################')

save_object = {
    'model_state_dict':model.state_dict(),
    'optimizer':optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'training_epoch': epoch,
    'valid_loss': train_loss/num_train_slices,
    'train_loss': valid_loss/num_valid_slices,
    'valid_dice': dice_score/num_dice_scores,
    'model_name': model_name,
    'eval_fold': EVAL_FOLD,
    'valid_fold': VALID_FOLD,
    'batch_size': BATCH_SIZE,
    'lr': LR,
    'epochs': EPOCHS,
    'data_augmentation': data_augmentation,
    'module_name': module_name,
    'means': dataset_means,
    'sdevs': dataset_sdevs,
    'train_losses': training_losses,
    'valid_losses': valid_losses,
    'valid_dices': dice_scores,
}
torch.save(save_object, RESULTS_FOLDER + model_name + '/final_model_epoch_' + str(epoch) +'_checkpoint.pth')
torch.save(model, RESULTS_FOLDER + model_name + '/final_model_epoch_' + str(epoch) +'_model.pth')
