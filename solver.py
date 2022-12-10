import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import os
import time
import datetime
import h5py
import numpy as np
import utils
from os.path import join
import lie_learn.spaces.S2 as S2
from model import SphericalGMMNet
from pdb import set_trace as st
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def eval(model, params, num_epochs=3, rotate=True):
    print("eval")
    
    
    s2_grids = utils.get_sphere_grids(b=params['bandwidth_0'], num_grids=params['num_grids'], base_radius=params['base_radius'])

    acc_overall = list()
    test_iterator = utils.load_data_h5(params, data_type="test", rotate=rotate, batch=False)
    for epoch in range(num_epochs):
        acc_all = []
        with torch.no_grad():
            for _, (inputs, labels) in enumerate(test_iterator):
                
                inputs = Variable(inputs)
                B, N, D = inputs.size()

                if inputs.shape[-1] == 2:
                    zero_padding = torch.zeros((B, N, 1), dtype=inputs.dtype)
                    inputs = torch.cat((inputs, zero_padding), -1)  # [B, N, 3]

                # Data Mapping
                inputs = utils.data_mapping(inputs, base_radius=params['base_radius'])  # [B, N, 3]
                inputs = utils.data_sphere_translation(inputs, s2_grids, params)

                outputs = model(inputs)
                outputs = torch.argmax(outputs, dim=-1)
                acc_all.append(np.mean(outputs.detach().cpu().numpy() == labels.numpy()))
            acc_overall.append(np.mean(np.array(acc_all)))
        print("[Epoch:" ,epoch , "Acc:" , np.mean(np.array(acc_all)) ,"\n")
                        
    return np.max(acc_overall)


def test(params, model_name, num_epochs=1000):


    print("Model Setting Up for test")
    # Model Configuration Setup
    model = SphericalGMMNet(params)
    model = model 
    if len(params['gpu'].split(",")) >= 2:
        model = nn.DataParallel(model)

    model_path = os.path.join(params['save_dir'], '{model_name}'.format(model_name=model_name))
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    # Generate the grids
    s2_grids = utils.get_sphere_grids(b=params['bandwidth_0'], num_grids=params['num_grids'], base_radius=params['base_radius'])

    test_iterator = utils.load_data_h5(params, data_type="test", rotate=True, batch=False)
    for epoch in range(num_epochs):
        acc_all = []
        with torch.no_grad():
            for _, (inputs, labels) in enumerate(test_iterator):
                inputs = Variable(inputs) 
                B, N, D = inputs.size()

                if inputs.shape[-1] == 2:
                    zero_padding = torch.zeros((B, N, 1), dtype=inputs.dtype) 
                    inputs = torch.cat((inputs, zero_padding), -1)  # [B, N, 3]

                # Data Mapping
                inputs = utils.data_mapping(inputs, base_radius=params['base_radius'])  # [B, N, 3]

                # Data Translation
                inputs = utils.data_sphere_translation(inputs, s2_grids, params)

                outputs = model(inputs)
                outputs = torch.argmax(outputs, dim=-1)
                acc_all.append(np.mean(outputs.detach().cpu().numpy() == labels.numpy()))

            print("[Epoch:" ,epoch , "Acc:" , np.mean(np.array(acc_all)) ,"\n")


def train(params):

    print("Loading Data")

    # # Load Data
    train_iterator = utils.load_data_h5(params, data_type="train")

    # # Model Setup
    print("Model Setting Up for train")


    model = SphericalGMMNet(params) 
    model = model 
    if len(params['gpu'].split(",")) >= 2:
        model = nn.DataParallel(model)

    # Model Configuration Setup
    optim = torch.optim.Adam(model.parameters(), lr=params['baselr'])
    cls_criterion = torch.nn.CrossEntropyLoss() 

    # Resume If Asked
    date_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if params['resume_training']:
        model_path = os.path.join(params['save_dir'], params['resume_training'])
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))


    # Generate the grids
    s2_grids = utils.get_sphere_grids(b=params['bandwidth_0'], num_grids=params['num_grids'], base_radius=params['base_radius'])

    # TODO [Visualize Grids]
    if params['visualize']:
        utils.visualize_sphere_grids(s2_grids, params)
    
    # Keep track of max Accuracy during training
    non_rotate_acc, rotate_acc = 0, 0
    max_non_rotate_acc, max_rotate_acc = 0, 0
    
    # Iterate by Epoch
    print("Start Training")
    for epoch in range(params['num_epochs']):
        print("epoch:")
        print(epoch)
        # Save the model for each step
        if non_rotate_acc > max_non_rotate_acc:
            max_non_rotate_acc = non_rotate_acc
            save_path = os.path.join(params['save_dir'], '{date_time}-NR-[{acc}]-model.ckpt'.format(date_time=date_time, acc=non_rotate_acc))
            torch.save(model.state_dict(), save_path)
        if rotate_acc > max_rotate_acc:
            max_rotate_acc = rotate_acc
            save_path = os.path.join(params['save_dir'], '{date_time}-R-[{acc}]-model.ckpt'.format(date_time=date_time, acc=rotate_acc))
            torch.save(model.state_dict(), save_path)

        # Running Model
        running_loss = []
        for batch_idx, (inputs, labels) in enumerate(train_iterator):
            """ Variable Setup """
            inputs, labels = Variable(inputs) , Variable(labels) 
            # print(inputs)
            # B, N, D = inputs.size()
            B, N, D= inputs.size()

            if inputs.shape[-1] == 2:
                zero_padding = torch.zeros((B, N, 1), dtype=inputs.dtype) 
                inputs = torch.cat((inputs, zero_padding), -1)  # [B, N, 3]

            # Data Mapping
            inputs = utils.data_mapping(inputs, base_radius=params['base_radius'])  # [B, N, 3]
            
            if params['visualize']:
                
                # TODO [Visualization [Raw]]
                origins = inputs.clone()
                # utils.visualize_raw(inputs, labels)
                
                # TODO [Visualization [Sphere]]
                print("---------- Static ------------")
                params['use_static_sigma'] = True
                inputs1 = utils.data_sphere_translation(inputs, s2_grids, params)  
                utils.visualize_sphere_sphere(origins, inputs1, labels, s2_grids, params, folder='sphere')
                
                print("\n---------- Covariance ------------")
                params['use_static_sigma'] = False
                params['sigma_layer_diff'] = False
                inputs2 = utils.data_sphere_translation(inputs, s2_grids, params)  
                utils.visualize_sphere_sphere(origins, inputs2, labels, s2_grids, params, folder='sphere')
                
                print("\n---------- Layer Diff ------------")
                params['use_static_sigma'] = False
                params['sigma_layer_diff'] = True
                inputs3 = utils.data_sphere_translation(inputs, s2_grids, params)  
                utils.visualize_sphere_sphere(origins, inputs3, labels, s2_grids, params, folder='sphere')
                return
            else:
                # Data Translation
                inputs = utils.data_sphere_translation(inputs, s2_grids, params) # list( list( Tensor([B, 2b, 2b]) * num_grids ) * num_centers)
            
            """ Run Model """
            # print("run model")
            outputs = model(inputs)

            """ Back Propagation """
            loss = cls_criterion(outputs, labels.squeeze())
            loss.backward(retain_graph=True)
            optim.step()
            running_loss.append(loss.item())

            print("Batch: ", batch_idx , "Epoch:" ,epoch , "Loss:" , np.mean(running_loss) ,"\n")


        non_rotate_acc = eval(model, params, rotate=False)
        print("not rotate accuracy")
        # print(non_rotate_acc)
        print("Batch: ", batch_idx , "Epoch:" ,epoch , "Loss:" , np.mean(running_loss), "Acc: ", non_rotate_acc  ,"\n")


        
        # rotate_acc = eval(model, params, rotate=True)
        # print(" rotate accuracy")
        # print(rotate_acc)
        # print("Batch: ", batch_idx , "Epoch:" ,epoch , "Loss:" , np.mean(running_loss), "Acc: ", rotate_acc  ,"\n")


    print("Finished Training")


if __name__ == '__main__':

    args = utils.load_args()

    params = {
        'train_dir': os.path.join(args.data_path, "train"),
        'test_dir' : os.path.join(args.data_path, "test"),
        'save_dir' : os.path.join('./', "save"),
        
        'gpu'       : args.gpu,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'num_points': args.num_points,
        'visualize' : bool(args.visualize),

        'log_interval': args.log_interval,
        'save_interval': args.save_interval,
        'baselr': args.baselr,
        'density_radius': args.density_radius,
 
        'rotate_deflection':  args.rotate_deflection,
        'num_grids':          args.num_grids,
        'base_radius':        args.base_radius,
        'static_sigma':       args.static_sigma,
        'use_static_sigma':   bool(args.use_static_sigma),
        'use_weights':        bool(args.use_weights),
        'sigma_layer_diff':   bool(args.sigma_layer_diff),

        'feature_out1': 8,
        'feature_out2': 16,
        'feature_out3': 32,
        'feature_out4': 64,
        'feature_out5': 128,

        'num_classes': args.num_classes,
        'num_so3_layers': args.num_so3_layers,

        'bandwidth_0': 10,
        'bandwidth_out1': 10,
        'bandwidth_out2': 8,
        'bandwidth_out3': 6,
        'bandwidth_out4': 4,
        'bandwidth_out5': 2,

        'resume_training': args.resume_training,
    }

    # os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']
    if args.resume_testing:
        test(params, args.resume_testing)
    else:
        train(params)

# END
