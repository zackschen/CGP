# 选取样本的进行映射的时候 只选预测正确的
from tkinter.messagebox import NO
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from torch.autograd import Variable
import sys
import torchvision
from torchvision import datasets, transforms
import os
import os.path
from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import pdb
import argparse,time
import math
from copy import deepcopy
matplotlib.use('Agg')
## Define ResNet18 model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)
            )
        self.act = OrderedDict()
        self.count = 0

    def forward(self, x):
        self.count = self.count % 2 
        self.act['conv_{}'.format(self.count)] = x.detach().cpu()
        self.count +=1
        out = relu(self.bn1(self.conv1(x)),inplace=True)
        self.count = self.count % 2 
        self.act['conv_{}'.format(self.count)] = out.detach().cpu()
        self.count +=1
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out,inplace=True)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, taskcla, nf):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1, 1)
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        
        self.taskcla = taskcla
        self.linear=torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.linear.append(nn.Linear(nf * 8 * block.expansion * 4, n, bias=False))
        self.act = OrderedDict()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        self.act['conv_in'] = x.view(bsz, 3, 32, 32).detach().cpu()
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))),inplace=True) 
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        feature = F.normalize(out, dim=1)
        y=[]
        for t,i in self.taskcla:
            y.append(self.linear[t](out))
        return y,feature

def ResNet18(taskcla, nf=32):
    return ResNet(BasicBlock, [2, 2, 2, 2], taskcla, nf)

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def adjust_learning_rate(optimizer, epoch, args):
    for param_group in optimizer.param_groups:
        if (epoch ==1):
            param_group['lr']=args.lr
        else:
            param_group['lr'] /= args.lr_factor  

def train(args, model, device, x,x_saugtrain,y, optimizer,criterion, task_id):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b]
        data,data_waug, target = data.to(device),x_saugtrain[b].to(device), y[b].to(device)
        bsz = target.shape[0]
        optimizer.zero_grad()        

        output,feature = model(data)
        loss = 0.0
        if args.supcon > 0.0:
            shuffle_idx = torch.randperm(bsz)
            mapping = {k:v for (v,k) in enumerate(shuffle_idx)}
            reverse_idx = torch.LongTensor([mapping[k] for k in sorted(mapping.keys())])     
            output_aug,feature_aug = model(data_waug[shuffle_idx])
            feature_aug = feature_aug[reverse_idx]

            sim_clean = torch.mm(feature, feature.t())
            mask = (torch.ones_like(sim_clean) - torch.eye(bsz, device=sim_clean.device)).bool()
            sim_clean = sim_clean.masked_select(mask).view(bsz, -1)

            sim_aug = torch.mm(feature, feature_aug.t())
            sim_aug = sim_aug.masked_select(mask).view(bsz, -1)   
            
            logits_pos = torch.bmm(feature.view(bsz,1,-1),feature_aug.view(bsz,-1,1)).squeeze(-1)
            logits_neg = torch.cat([sim_clean,sim_aug],dim=1)

            logits = torch.cat([logits_pos,logits_neg],dim=1)
            instance_labels = torch.zeros(bsz).long().cuda()
            
            loss_instance = criterion(logits/args.temp, instance_labels)  
            loss += loss_instance * args.supcon 

        celoss = criterion(output[task_id], target)
        loss += celoss

        loss.backward()
        optimizer.step()

def train_projected(args,model,device,x,x_saugtrain,y,optimizer,criterion,feature_mat,task_id):
    model.train()
    # Loop batches
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b]
        data,data_waug, target = data.to(device),x_saugtrain[b].to(device), y[b].to(device)
        bsz = target.shape[0]
        optimizer.zero_grad()    

        output,feature = model(data)
        loss = 0.0
        if args.supcon > 0.0:
            shuffle_idx = torch.randperm(bsz)
            mapping = {k:v for (v,k) in enumerate(shuffle_idx)}
            reverse_idx = torch.LongTensor([mapping[k] for k in sorted(mapping.keys())])     
            output_aug,feature_aug = model(data_waug[shuffle_idx])
            feature_aug = feature_aug[reverse_idx]

            sim_clean = torch.mm(feature, feature.t())
            mask = (torch.ones_like(sim_clean) - torch.eye(bsz, device=sim_clean.device)).bool()
            sim_clean = sim_clean.masked_select(mask).view(bsz, -1)

            sim_aug = torch.mm(feature, feature_aug.t())
            sim_aug = sim_aug.masked_select(mask).view(bsz, -1)   
            
            logits_pos = torch.bmm(feature.view(bsz,1,-1),feature_aug.view(bsz,-1,1)).squeeze(-1)
            logits_neg = torch.cat([sim_clean,sim_aug],dim=1)

            logits = torch.cat([logits_pos,logits_neg],dim=1)
            instance_labels = torch.zeros(bsz).long().cuda()
            
            loss_instance = criterion(logits/args.temp, instance_labels)   
            loss += loss_instance * args.supcon

        celoss = criterion(output[task_id], target)
        loss += celoss        
        loss.backward()
        # Gradient Projections 
        kk = 0 
        for k, (m,params) in enumerate(model.named_parameters()):
            if len(params.size())==4:
                sz =  params.grad.data.size(0)

                feature_class_mat = feature_mat[kk]
                for i in range(len(feature_class_mat)):
                    params.grad.data = params.grad.data - (torch.mm(params.grad.data.view(sz,-1),\
                                                            feature_class_mat[i])).view(params.size())

                kk+=1
            elif len(params.size())==1 and task_id !=0:
                params.grad.data.fill_(0)

        optimizer.step()

def test(args, model, device, x,y, criterion, task_id):
    model.eval()
    total_loss = 0
    total_num = 0 
    correct = 0
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    with torch.no_grad():
        # Loop batches
        for i in range(0,len(r),args.batch_size_test):
            if i+args.batch_size_test<=len(r): b=r[i:i+args.batch_size_test]
            else: b=r[i:]
            data = x[b]
            data, target = data.to(device), y[b].to(device)
            output,_ = model(data)
            loss = criterion(output[task_id], target)
            pred = output[task_id].argmax(dim=1, keepdim=True) 
            
            correct    += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*len(b)
            total_num  += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc

def update_GPM (model, taskId, mat_list, threshold, feature_list=[], similarities = [], proto_index_dict = {}):
    print ('Threshold: ', threshold) 
    if not feature_list:
        # After First Task 
        for i in range(len(mat_list)):
            similarity_sub = similarities[i]
            feature_class_list = []
            activation = mat_list[i]
            proto_class_dict = {}
            proto_class_index_dict = {}
            # criteria (Eq-5)
            for j in range(len(activation)):
                activation_task = activation[j]
                if j > 0:
                    similarity = similarity_sub[j]
                    simi_tasks = [task for task,simi in similarity.items() if np.abs(simi) > model.simi_threshold]
                    if len(simi_tasks) > 0:
                        for task in simi_tasks:
                            # origin_task = proto_class_dict[task]
                            # proto_class_dict[j] = origin_task
                            if not task in proto_class_index_dict.keys():
                                continue
                            index = proto_class_index_dict[task]
                            proto_class_index_dict[j] = index

                            activation_task_orin = np.hstack((activation_task,activation[task]))
                        
                            U1,S1,Vh1 = np.linalg.svd(activation_task_orin, full_matrices=False)
                            sval_total = (S1**2).sum()
                            sval_ratio = (S1**2)/sval_total
                            r = np.sum(np.cumsum(sval_ratio)<threshold[i]) 
                            feature_class_list[index] = U1[:,0:r]
                    else:
                        U1,S1,Vh1 = np.linalg.svd(activation_task, full_matrices=False)
                        sval_total = (S1**2).sum()
                        for feature in feature_class_list:
                            activation_task_hat = activation_task - np.dot(np.dot(feature,feature.transpose()),activation_task)
                        U,S,Vh = np.linalg.svd(activation_task_hat, full_matrices=False)

                        # criteria (Eq-9)
                        sval_hat = (S**2).sum()
                        sval_ratio = (S**2)/sval_total               
                        accumulated_sval = (sval_total-sval_hat)/sval_total
                        
                        r = 0
                        for ii in range (sval_ratio.shape[0]):
                            if accumulated_sval < threshold[i]:
                                accumulated_sval += sval_ratio[ii]
                                r += 1
                            else:
                                break
                        if r == 0:
                            print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                            continue
                        feature_class_list.append(U[:,0:r])
                        proto_class_index_dict[j] = len(feature_class_list)-1
                else:
                    U,S,Vh = np.linalg.svd(activation_task, full_matrices=False)
                    # proto_class_dict[j] = j
                    proto_class_index_dict[j] = j
                    sval_total = (S**2).sum()
                    sval_ratio = (S**2)/sval_total
                    r = np.sum(np.cumsum(sval_ratio)<threshold[i]) #+1  求k近似，k个基的和达到了threshold
                    feature_class_list.append(U[:,0:r])
            
            # proto_simi_class_dict[i] = proto_class_dict
            proto_index_dict[i] = proto_class_index_dict
            feature_list.append(feature_class_list)
    else:
        for i in range(len(mat_list)):
            # proto_simi_class = proto_simi_class_dict[i] 
            proto_class_index_dict = proto_index_dict[i] 

            similarity_sub = similarities[i]

            feature_class_list = feature_list[i]
            activation = mat_list[i]
            pre_class_count = taskId*len(activation)
            for task_id in range(len(activation)):
                activation_task = activation[task_id]

                similarity = similarity_sub[task_id]
                simi_tasks = [task for task,simi in similarity.items() if np.abs(simi) > model.simi_threshold]
                if len(simi_tasks) > 0:
                    U,S,Vh=np.linalg.svd(activation_task, full_matrices=False)
                    for task in simi_tasks:
                        if not task in proto_class_index_dict.keys():
                                continue
                        index = proto_class_index_dict[task]
                        proto_class_index_dict[pre_class_count+task_id] = index

                        similarity_task_feature = feature_class_list[index]
                        theta = np.dot(np.dot(np.dot(similarity_task_feature.transpose(),activation_task),activation_task.transpose()),similarity_task_feature)
                        sval_total = (S**2).sum()
                        # Projected Representation (Eq-8)
                        act_hat = activation_task - np.dot(np.dot(similarity_task_feature,similarity_task_feature.transpose()),activation_task)
                        U1,S1,Vh1 = np.linalg.svd(act_hat, full_matrices=False)

                        S_to_U = {}
                        S_from = []
                        S_New = np.array([])
                        for j in range(theta.shape[0]):
                            S_New = np.append(S_New,theta[j][j])
                            S_to_U[theta[j][j]] = similarity_task_feature[:,j]
                            S_from.append(theta[j][j])
                        for s in range(len(S1)):
                            S_New = np.append(S_New,S1[s]*S1[s])
                            S_to_U[S1[s]*S1[s]] = U1[:,s]
                        S_New = np.flip(np.sort(S_New))

                        # criteria (Eq-9)
                        sval_hat = (S1**2).sum()
                        sval_ratio = (S1**2)/sval_total               
                        accumulated_sval = (sval_total-sval_hat)/sval_total
                        
                        r = 0
                        for ii in range (sval_ratio.shape[0]):
                            if accumulated_sval < threshold[i]:
                                accumulated_sval += sval_ratio[ii]
                                r += 1
                            else:
                                break
                        if r == 0:
                            print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                            continue

                        sigular = S_New[:r]
                        Ui = feature_list[i][index]
                        for s in sigular:
                            if not s in S_from:
                                Ui=np.hstack((Ui,S_to_U[s].reshape(-1,1)))
                        # update GPM
                        if Ui.shape[1] > Ui.shape[0] :
                            feature_list[i][index]=Ui[:,0:Ui.shape[0]]
                        else:
                            feature_list[i][index]=Ui
                else:
                    U1,S1,Vh1 = np.linalg.svd(activation_task, full_matrices=False)
                    sval_total = (S1**2).sum()
                    for feature in feature_class_list:
                        activation_task_hat = activation_task - np.dot(np.dot(feature,feature.transpose()),activation_task)
                    U,S,Vh = np.linalg.svd(activation_task_hat, full_matrices=False)

                    # criteria (Eq-9)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2)/sval_total               
                    accumulated_sval = (sval_total-sval_hat)/sval_total
                    
                    r = 0
                    for ii in range (sval_ratio.shape[0]):
                        if accumulated_sval < threshold[i]:
                            accumulated_sval += sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                        continue
                    feature_class_list.append(U[:,0:r])
                    proto_class_index_dict[pre_class_count+task_id] = len(feature_class_list)-1

    print('-'*40)
    print('Gradient Constraints Summary')
    print('-'*40)
    for i in range(len(feature_list)):
        print ('Layer {} : {}/{}'.format(i+1,feature_list[i][0].shape[1], feature_list[i][0].shape[0]))
    print('-'*40)
    return feature_list,proto_index_dict

def computeProto(model, x,y, task_id, threshold, proto_list_dict = {}):
    batch_list  = [10,10,10,10,10,10,10,10,50,50,50,100,100,100,100,100,100] #scaled
    # network arch 
    stride_list = [1, 1,1,1,1, 2,1,1,1, 2,1,1,1, 2,1,1,1]
    map_list    = [32, 32,32,32,32, 32,16,16,16, 16,8,8,8, 8,4,4,4] 
    in_channel  = [ 3, 20,20,20,20, 20,40,40,40, 40,80,80,80, 80,160,160,160] 

    pad = 1
    sc_list=[5,9,13]
    p1d = ((0,0),(0,0),(1,1),(1,1))

    features = None
    labels = np.array([])
    features_all = {}
    continue_array = [0]*args.class_per_task

    model.eval()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).cuda()
    with torch.no_grad():
        # Loop batches
        for i in range(0,len(r),args.batch_size_train):
            if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
            else: b=r[i:]
            data = x[b]
            data, target = data.cuda(), y[b].cuda()

            output,_ = model(data)
            
            pred = output[task_id].argmax(dim=1, keepdim=True) 
            index_right,_ = torch.where(pred == target.view_as(pred))

            for target_i in range(args.class_per_task):
                index_i = torch.where(pred == target_i)[0]
                index = [i.item() for i in index_i if i in index_right]
                
                if not target_i in features_all.keys():
                    features = {0:model.act['conv_in'][index].numpy(), 
                                1:model.layer1[0].act['conv_0'][index].numpy(),
                                2:model.layer1[0].act['conv_1'][index].numpy(), 
                                3:model.layer1[1].act['conv_0'][index].numpy(),
                                4:model.layer1[1].act['conv_1'][index].numpy(),
                                5:model.layer2[0].act['conv_0'][index].numpy(), 
                                6:model.layer2[0].act['conv_1'][index].numpy(), 
                                7:model.layer2[1].act['conv_0'][index].numpy(), 
                                8:model.layer2[1].act['conv_1'][index].numpy(),
                                9:model.layer3[0].act['conv_0'][index].numpy(),
                                10:model.layer3[0].act['conv_1'][index].numpy(),
                                11:model.layer3[1].act['conv_0'][index].numpy(),
                                12:model.layer3[1].act['conv_1'][index].numpy(),
                                13:model.layer4[0].act['conv_0'][index].numpy(), 
                                14:model.layer4[0].act['conv_1'][index].numpy(), 
                                15:model.layer4[1].act['conv_0'][index].numpy(), 
                                16:model.layer4[1].act['conv_1'][index].numpy()}
                    features_all[target_i] = features
                else:
                    features = features_all[target_i]
                    features[0] = np.concatenate((features[0], model.act['conv_in'][index].numpy()),axis=0)
                    features[1] = np.concatenate((features[1], model.layer1[0].act['conv_0'][index].numpy()),axis=0)
                    features[2] = np.concatenate((features[2], model.layer1[0].act['conv_1'][index].numpy()),axis=0)
                    features[3] = np.concatenate((features[3], model.layer1[1].act['conv_0'][index].numpy()),axis=0)
                    features[4] = np.concatenate((features[4], model.layer1[1].act['conv_1'][index].numpy()),axis=0)
                    features[5] = np.concatenate((features[5], model.layer2[0].act['conv_0'][index].numpy()),axis=0)
                    features[6] = np.concatenate((features[6], model.layer2[0].act['conv_1'][index].numpy()),axis=0)
                    features[7] = np.concatenate((features[7], model.layer2[1].act['conv_0'][index].numpy()),axis=0)
                    features[8] = np.concatenate((features[8], model.layer2[1].act['conv_1'][index].numpy()),axis=0)
                    features[9] = np.concatenate((features[9], model.layer3[0].act['conv_0'][index].numpy()),axis=0)
                    features[10] = np.concatenate((features[10], model.layer3[0].act['conv_1'][index].numpy()),axis=0)
                    features[11] = np.concatenate((features[11], model.layer3[1].act['conv_0'][index].numpy()),axis=0)
                    features[12] = np.concatenate((features[12], model.layer3[1].act['conv_1'][index].numpy()),axis=0)
                    features[13] = np.concatenate((features[13], model.layer4[0].act['conv_0'][index].numpy()),axis=0)
                    features[14] = np.concatenate((features[14], model.layer4[0].act['conv_1'][index].numpy()),axis=0)
                    features[15] = np.concatenate((features[15], model.layer4[1].act['conv_0'][index].numpy()),axis=0)
                    features[16] = np.concatenate((features[16], model.layer4[1].act['conv_1'][index].numpy()),axis=0)

                if len(features[0]) >= batch_list[-1]:
                    features_copy = {}
                    for key,value in enumerate(features):
                        features_copy[key] = features[key][:batch_list[-1]]
                    continue_array[target_i] = 1

                labels = np.hstack((labels,target[index].cpu().numpy()))

            if not 0 in continue_array:
                break
    
    labels_set = np.unique(labels)

    mat_final=[]
    mat_list=[]
    mat_sc_list=[]
    similarity_all = []
    for i in features.keys():
        mat_class_list = []
        mat_sc_class_list=[]

        if not i in proto_list_dict.keys():
            proto_list_dict[i] = []
        proto_class_list = proto_list_dict[i]

        similarities = []

        for item in labels_set:
        
            feature = features_all[item]
            feature_classwise = feature[i]
            proto = np.mean(feature_classwise, axis=0)

            if task_id == 0 and item == 0:
                similarities.append(-1)
            else:
                similarity = compute_similarity_with_proto(model,proto,proto_class_list)
                similarities.append(similarity)
            proto_class_list.append(proto)
            
            if i==0:
                ksz = 3
            else:
                ksz = 3 
            bsz=batch_list[i]
            bsz = min(bsz,feature_classwise.shape[0])
            st = stride_list[i]     
            k=0
            s=compute_conv_output_size(map_list[i],ksz,stride_list[i],pad)
            mat = np.zeros((ksz*ksz*in_channel[i],s*s*bsz))
            act = np.pad(feature_classwise, p1d, "constant")
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        mat[:,k]=act[kk,:,st*ii:ksz+st*ii,st*jj:ksz+st*jj].reshape(-1)
                        k +=1
            mat_class_list.append(mat)
            # For Shortcut Connection
            if i in sc_list:
                k=0
                s=compute_conv_output_size(map_list[i],1,stride_list[i])
                mat = np.zeros((1*1*in_channel[i],s*s*bsz))
                act = feature_classwise
                for kk in range(bsz):
                    for ii in range(s):
                        for jj in range(s):
                            mat[:,k]=act[kk,:,st*ii:1+st*ii,st*jj:1+st*jj].reshape(-1)
                            k +=1
                mat_sc_class_list.append(mat) 
                

        mat_list.append(mat_class_list)
        if i in sc_list:
            mat_sc_list.append(mat_sc_class_list)

        proto_list_dict[i] = proto_class_list
        similarity_all.append(similarities)

    ik=0
    similarity_all_new = []
    proto_list_dict_new = {}
    for j in range (len(mat_list)):
        mat_final.append(mat_list[j])
        similarity_all_new.append(similarity_all[j])
        proto_list_dict_new[j+ik] = proto_list_dict[j]
        if j in [6,10,14]:
            similarity_all_new.append(similarity_all[j])
            mat_final.append(mat_sc_list[ik])
            ik+=1
            proto_list_dict_new[j+ik] = proto_list_dict[j]
    
    print('-'*30)
    print('Representation Matrix')
    print('-'*30)
    for i in range(len(mat_final)):
        print ('Layer {} : {}'.format(i+1,mat_final[i][0].shape))
    print('-'*30)

    return mat_final, proto_list_dict, similarity_all_new

def compute_similarity_with_proto(model,current_proto,task_protos):
    similarity_dict = {}
    for k in range(len(task_protos)):
        current = current_proto.reshape(-1)
        task = task_protos[k].reshape(-1)
        similarity = np.dot(current,task)/(np.linalg.norm(current) * np.linalg.norm(task))
        similarity_dict[k] = similarity
    return similarity_dict

def seed_torch(seed=1):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True


def main(args):
    tstart=time.time()
    ## Device Setting 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_torch(args.seed)

    from dataloader import five_datasets as data_loader
    data,taskcla,inputsize=data_loader.get(pc_valid=args.pc_valid)

    acc_matrix=np.zeros((5,5))
    criterion = torch.nn.CrossEntropyLoss()

    task_id = 0
    task_list = []

    proto_list_dict = {}
    similarities = None

    for k,ncla in taskcla:
        threshold = np.array([0.965] * 20)
     
        print('*'*100)
        print('Task {:2d} ({:s})'.format(k,data[k]['name']))
        print('*'*100)
        xtrain=data[k]['train']['x']
        ytrain=data[k]['train']['y']
        x_saugtrain=data[k]['train']['saug']
        xvalid=data[k]['valid']['x']
        yvalid=data[k]['valid']['y']
        xtest =data[k]['test']['x']
        ytest =data[k]['test']['y']

        task_list.append(k)

        lr = args.lr 
        best_loss=np.inf
        print ('-'*40)
        print ('Task ID :{} | Learning Rate : {}'.format(task_id, lr))
        print ('-'*40)
        
        if task_id==0:
            model = ResNet18(taskcla,20).to(device)
            model.simi_threshold = args.simi_threshold
            best_model=get_model(model)
            feature_list =[]
            feature_list_origin =[]
            optimizer = optim.SGD(model.parameters(), lr=lr)

            for epoch in range(1, args.n_epochs+1):
                # Train
                clock0=time.time()
                train(args, model, device, xtrain,x_saugtrain, ytrain, optimizer, criterion, k)
                clock1=time.time()
                tr_loss,tr_acc = test(args, model, device, xtrain, ytrain,  criterion, k)
                print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                            tr_loss,tr_acc, 1000*(clock1-clock0)),end='')
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid, yvalid,  criterion, k)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),end='')
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=get_model(model)
                    patience=args.lr_patience
                    print(' *',end='')
                else:
                    patience-=1
                    if patience<=0:
                        lr/=args.lr_factor
                        print(' lr={:.1e}'.format(lr),end='')
                        if lr<args.lr_min:
                            print()
                            break
                        patience=args.lr_patience
                        adjust_learning_rate(optimizer, epoch, args)
                print()
            set_model_(model,best_model)
            # Test
            print ('-'*40)
            test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion, k)
            print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))
            # Memory Update  
            mat_list, proto_list_dict, similarities = computeProto(model,xtrain, ytrain,task_id,threshold,proto_list_dict)
            feature_list,proto_index_dict = update_GPM(model, task_id, mat_list, threshold, feature_list, similarities)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
            feature_mat = []
            # Projection Matrix Precomputation
            for i in range(len(feature_list)):
                feature_class_mat = []
                feature_class_list = feature_list[i]
                for feature in feature_class_list:
                    Uf=torch.Tensor(np.dot(feature,feature.transpose())).to(device)
                    feature_class_mat.append(Uf)
                feature_mat.append(feature_class_mat)
            print ('-'*40)

            for epoch in range(1, args.n_epochs+1):
                # Train 
                clock0=time.time()
                train_projected(args, model,device,xtrain,x_saugtrain, ytrain,optimizer,criterion,feature_mat,k)
                clock1=time.time()
                tr_loss, tr_acc = test(args, model, device, xtrain, ytrain,criterion,k)
                print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                        tr_loss, tr_acc, 1000*(clock1-clock0)),end='')
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid, yvalid, criterion,k)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),end='')
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=get_model(model)
                    patience=args.lr_patience
                    print(' *',end='')
                else:
                    patience-=1
                    if patience<=0:
                        lr/=args.lr_factor
                        print(' lr={:.1e}'.format(lr),end='')
                        if lr<args.lr_min:
                            print()
                            break
                        patience=args.lr_patience
                        adjust_learning_rate(optimizer, epoch, args)
                print()
            set_model_(model,best_model)
            # Test 
            test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion,k)
            print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))  
            # Memory Update 
            mat_list, proto_list_dict, similarities = computeProto(model,xtrain, ytrain,task_id,threshold,proto_list_dict)
            feature_list,proto_index_dict = update_GPM (model, task_id, mat_list, threshold, feature_list, similarities, proto_index_dict)
        # save accuracy 
        jj = 0 
        for ii in np.array(task_list)[0:task_id+1]:
            xtest =data[ii]['test']['x']
            ytest =data[ii]['test']['y'] 
            _, acc_matrix[task_id,jj] = test(args, model, device, xtest, ytest,criterion,ii) 
            jj +=1
        print('Accuracies =')
        for i_a in range(task_id+1):
            print('\t',end='')
            for j_a in range(acc_matrix.shape[1]):
                print('{:5.1f}% '.format(acc_matrix[i_a,j_a]),end='')
            print()
        # update task id 
        task_id +=1
    print('-'*50)
    # Simulation Results 
    print ('Task Order : {}'.format(np.array(task_list)))
    print ('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[-1].mean())) 
    bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1]) 
    print ('Backward transfer: {:5.2f}%'.format(bwt))
    print('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000))
    print('-'*50)


if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='5 datasets with GPM')
    parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=100, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=37, metavar='S',
                        help='random seed (default: 37)')
    parser.add_argument('--pc_valid',default=0.05,type=float,
                        help='fraction of training data used for validation')
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.08, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr_min', type=float, default=1e-3, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=5, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=3, metavar='LRF',
                        help='lr decay factor (default: 2)')
    parser.add_argument('--add_description', default='_')
    parser.add_argument('--simi_threshold', type=float, default=0.5)
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    parser.add_argument('--supcon', type=float, default=0.1,
                        help='temperature for loss function')

    args = parser.parse_args()
    args.class_per_task = 10
    
    log_path = 'checkpoints'
    exp_name = f'Five/5tasks_{args.batch_size_train}batch'
    exp_name += f'_{args.add_description}_T_{args.simi_threshold}_lr_{args.lr}'
    args.model_path = os.path.join(log_path, exp_name)
    os.makedirs(args.model_path, exist_ok=True)
    ## create checkpoint dir and copy files
    file_dir = os.path.join(args.model_path, 'files')
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(file_dir, exist_ok=True)
    file_name = os.path.basename(__file__)
    os.system(f'cp {file_name} {file_dir}')
    ## logger, copy stdout to a file
    from logger import FileOutputDuplicator
    sys.stdout = FileOutputDuplicator(sys.stdout, os.path.join(args.model_path, 'log.txt'), 'w')

    print('='*100)
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('='*100)
    
    main(args)



