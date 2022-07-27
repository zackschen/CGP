import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import os.path
from collections import OrderedDict
import matplotlib
import numpy as np
import random
import argparse,time
from copy import deepcopy
from dataloader import cifar100 as cf100

## Define AlexNet model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class AlexNet(nn.Module):
    def __init__(self,taskcla):
        super(AlexNet, self).__init__()
        self.act=OrderedDict()
        self.map =[]
        self.ksize=[]
        self.in_channel =[]
        self.map.append(32)
        self.conv1 = nn.Conv2d(3, 64, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        s=compute_conv_output_size(32,4)
        s=s//2
        self.ksize.append(4)
        self.in_channel.append(3)
        self.map.append(s)
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False)
        s=compute_conv_output_size(s,3)
        s=s//2
        self.ksize.append(3)
        self.in_channel.append(64)
        self.map.append(s)
        self.conv3 = nn.Conv2d(128, 256, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(256, track_running_stats=False)
        s=compute_conv_output_size(s,2)
        s=s//2
        self.smid=s
        self.ksize.append(2)
        self.in_channel.append(128)
        self.map.append(256*self.smid*self.smid)
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)

        self.fc1 = nn.Linear(256*self.smid*self.smid,2048, bias=False)
        self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.fc2 = nn.Linear(2048,2048, bias=False)
        self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.map.extend([2048])
        
        self.taskcla = taskcla
        self.fc3=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.fc3.append(torch.nn.Linear(2048,n,bias=False))
        
    def forward(self, x):
        bsz = deepcopy(x.size(0))
        self.act['conv1']=x
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

        self.act['conv2']=x
        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

        self.act['conv3']=x
        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

        x=x.view(bsz,-1)
        self.act['fc1']=x
        x = self.fc1(x)
        x = self.drop2(self.relu(self.bn4(x)))

        self.act['fc2']=x  
        x = self.fc2(x)
        x = self.drop2(self.relu(self.bn5(x)))
        feature = F.normalize(x, dim=1)

        y=[]
        for t,i in self.taskcla:
            y.append(self.fc3[t](x))
            
        return y,feature

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

def train(args,epoch, model, device, x,x_waugtrain,x_saugtrain,y, optimizer,criterion, task_id):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data,data_waug,data_saug = x[b],x_waugtrain[b],x_saugtrain[b]
        data,data_waug,data_saug, target = data.to(device), data_waug.to(device),data_saug.to(device), y[b].to(device)
        bsz = target.shape[0]
        optimizer.zero_grad()  

        output,feature = model(data)


        loss = 0.0
        if args.supcon > 0.0:
            shuffle_idx = torch.randperm(bsz)
            mapping = {k:v for (v,k) in enumerate(shuffle_idx)}
            reverse_idx = torch.LongTensor([mapping[k] for k in sorted(mapping.keys())])     
            output_aug,feature_aug = model(data_saug[shuffle_idx])
            feature_aug = feature_aug[reverse_idx]

            sim_clean = torch.mm(feature, feature.t())
            mask = (torch.ones_like(sim_clean) - torch.eye(sim_clean.shape[0], device=sim_clean.device)).bool()
            sim_clean = sim_clean.masked_select(mask).view(sim_clean.shape[0], -1)

            sim_aug = torch.mm(feature, feature_aug.t())
            sim_aug = sim_aug.masked_select(mask).view(sim_clean.shape[0], -1)   
            
            logits_pos = torch.bmm(feature.view(sim_clean.shape[0],1,-1),feature_aug.view(sim_clean.shape[0],-1,1)).squeeze(-1)
            logits_neg = torch.cat([sim_clean,sim_aug],dim=1)

            logits = torch.cat([logits_pos,logits_neg],dim=1)
            instance_labels = torch.zeros(sim_clean.shape[0]).long().cuda()
            
            loss_instance = criterion(logits/args.temp, instance_labels) 
            loss += loss_instance * args.supcon

        celoss = criterion(output[task_id], target)
        loss += celoss

        loss.backward()
        optimizer.step()

def train_projected(args,epoch, model,device, x,x_waugtrain,x_saugtrain,y,optimizer,criterion,feature_mat,task_id):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data,data_waug,data_saug = x[b],x_waugtrain[b],x_saugtrain[b]
        data,data_waug,data_saug, target = data.to(device), data_waug.to(device),data_saug.to(device), y[b].to(device)
        bsz = target.shape[0]
        optimizer.zero_grad()  

        output,feature = model(data)

        loss = 0.0
        if args.supcon > 0.0:
            shuffle_idx = torch.randperm(bsz)
            mapping = {k:v for (v,k) in enumerate(shuffle_idx)}
            reverse_idx = torch.LongTensor([mapping[k] for k in sorted(mapping.keys())])     
            _,feature_aug = model(data_saug[shuffle_idx])
            feature_aug = feature_aug[reverse_idx]

            sim_clean = torch.mm(feature, feature.t())
            mask = (torch.ones_like(sim_clean) - torch.eye(sim_clean.shape[0], device=sim_clean.device)).bool()
            sim_clean = sim_clean.masked_select(mask).view(sim_clean.shape[0], -1)

            sim_aug = torch.mm(feature, feature_aug.t())
            sim_aug = sim_aug.masked_select(mask).view(sim_clean.shape[0], -1)   
            
            logits_pos = torch.bmm(feature.view(sim_clean.shape[0],1,-1),feature_aug.view(sim_clean.shape[0],-1,1)).squeeze(-1)
            logits_neg = torch.cat([sim_clean,sim_aug],dim=1)

            logits = torch.cat([logits_pos,logits_neg],dim=1)
            instance_labels = torch.zeros(sim_clean.shape[0]).long().cuda()
            
            loss_instance = criterion(logits/args.temp, instance_labels) 
            loss += loss_instance * args.supcon

        celoss = criterion(output[task_id], target)
        loss += celoss
        loss.backward()

        kk = 0 
        for k, (m,params) in enumerate(model.named_parameters()):
            if k<15 and len(params.size())!=1:
                sz =  params.grad.data.size(0)
                feature_class_mat = feature_mat[kk]
                for i in range(len(feature_class_mat)):
                    params.grad.data = params.grad.data - (torch.mm(params.grad.data.view(sz,-1),\
                                                            feature_class_mat[i])).view(params.size())
                kk +=1
            elif (k<15 and len(params.size())==1) and task_id !=0 :
                params.grad.data.fill_(0)
        optimizer.step()

def test(args, model, device,  x,y, criterion, task_id):
    model.eval()
    total_loss = 0
    total_num = 0 
    correct = 0
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    with torch.no_grad():
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
        for i in range(len(mat_list)):
            similarity_sub = similarities[i]
            feature_class_list = []
            activation = mat_list[i]
            proto_class_index_dict = {}
            for j in range(len(activation)):
                activation_task = activation[j]
                if j > 0:
                    similarity = similarity_sub[j]
                    simi_tasks = [task for task,simi in similarity.items() if np.abs(simi) > model.simi_threshold]
                    if len(simi_tasks) > 0:
                        for task in simi_tasks:
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
                    proto_class_index_dict[j] = j
                    sval_total = (S**2).sum()
                    sval_ratio = (S**2)/sval_total
                    r = np.sum(np.cumsum(sval_ratio)<threshold[i])
                    feature_class_list.append(U[:,0:r])
            
            proto_index_dict[i] = proto_class_index_dict
            feature_list.append(feature_class_list)
    else:
        for i in range(len(mat_list)):
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
    batch_list=[2*12,100,100,125,125] 
    act_keys = list(model.act.keys())
    mat_list=[]
    similarity_all = []
    features_all = {}
    continue_array = [0]*10
    labels = np.array([])
    model.eval()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).cuda()
    with torch.no_grad():
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

                for i in range(5):
                    act_key = act_keys[i]
                    if not i in features_all.keys():
                        features_all[i] = {}

                    if target_i in features_all[i].keys():
                        features_all[i][target_i] = np.concatenate((features_all[i][target_i],model.act[act_key][index].detach().cpu().numpy()))
                        if len(features_all[i][target_i]) >= batch_list[-1]:
                            features_all[i][target_i] = features_all[i][target_i][:batch_list[-1]]
                            continue_array[target_i] = 1
                    else:
                        features_all[i][target_i] = model.act[act_key][index].detach().cpu().numpy()
            
                labels = np.hstack((labels,target[index].cpu().numpy()))
            
            if not 0 in continue_array:
                break

    labels_set = np.unique(labels)
    labels = np.array(labels)
    for i in range(len(act_keys)):
        act_key = act_keys[i]
        similarities = []

        features = features_all[i]

        bsz=batch_list[i]

        mat_class = []
        if not i in proto_list_dict.keys():
            proto_list_dict[i] = []
        proto_class_list = proto_list_dict[i]

        for item in labels_set:
            k=0
            feature_classwise = features[item]
            bsz = min(bsz,feature_classwise.shape[0])
            proto = np.mean(feature_classwise, axis=0)

            if task_id == 0 and item == 0:
                similarities.append(-1)
            else:
                similarity = compute_similarity_with_proto(model,proto,proto_class_list)
                similarities.append(similarity)
            proto_class_list.append(proto)

            if i<3:
                ksz= model.ksize[i]
                s=compute_conv_output_size(model.map[i],model.ksize[i])
                mat = np.zeros((model.ksize[i]*model.ksize[i]*model.in_channel[i],s*s*bsz))
                act = feature_classwise
                for kk in range(bsz):
                    for ii in range(s):
                        for jj in range(s):
                            mat[:,k]=act[kk,:,ii:ksz+ii,jj:ksz+jj].reshape(-1) 
                            k +=1
                mat_class.append(mat)
            else:
                act = feature_classwise
                activation = act[0:bsz].transpose()
                mat_class.append(activation)

        proto_list_dict[i] = proto_class_list
        similarity_all.append(similarities)
        mat_list.append(mat_class)

    return mat_list, proto_list_dict, similarity_all

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_torch(args.seed)

    data,taskcla,inputsize=cf100.get(seed=args.seed, pc_valid=args.pc_valid, num_task = args.num_task)

    acc_matrix=np.zeros((10,10))
    criterion = torch.nn.CrossEntropyLoss()

    task_id = 0
    task_list = []

    proto_list_dict = {}
    similarities = None


    for k,ncla in taskcla:
        threshold = np.array([0.97] * 5) + task_id*np.array([0.003] * 5)
     
        print('*'*100)
        print('Task {:2d} ({:s})'.format(k,data[k]['name']))
        print('*'*100)
        xtrain=data[k]['train']['x']
        x_waugtrain=data[k]['train']['waug']
        x_saugtrain=data[k]['train']['saug']
        ytrain=data[k]['train']['y']
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
            model = AlexNet(taskcla).to(device)
            model.simi_threshold = args.simi_threshold
            print ('Model parameters ---')
            for k_t, (m, param) in enumerate(model.named_parameters()):
                print (k_t,m,param.shape)
            print ('-'*40)

            best_model=get_model(model)
            feature_list =[]
            optimizer = optim.SGD(model.parameters(), lr=lr)

            for epoch in range(1, args.n_epochs+1):
                # Train
                clock0=time.time()
                train(args,epoch, model, device, xtrain,x_waugtrain, x_saugtrain, ytrain, optimizer, criterion, k)
                clock1=time.time()
                tr_loss,tr_acc = test(args, model, device, xtrain, ytrain,  criterion, k)
                print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms, lr {:.2f}'.format(epoch,\
                                                            tr_loss,tr_acc, 1000*(clock1-clock0), optimizer.param_groups[0]['lr']),end='')
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid,yvalid,  criterion, k)
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
            test_loss, test_acc = test(args, model, device, xtest,ytest,  criterion, k)
            print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))
            mat_list, proto_list_dict, similarities = computeProto(model,xtrain, ytrain,task_id,threshold,proto_list_dict)
            feature_list,proto_index_dict = update_GPM(model, task_id, mat_list, threshold, feature_list, similarities)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
            feature_mat = []
            # Projection Matrix Precomputation
            for i in range(len(model.act)):
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
                train_projected(args, epoch, model,device,xtrain,x_waugtrain,x_saugtrain, ytrain,optimizer,criterion,feature_mat,k)
                clock1=time.time()
                tr_loss,tr_acc = test(args, model, device, xtrain, ytrain,  criterion, k)
                print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms , lr {:.2f}'.format(epoch,\
                                                            tr_loss,tr_acc, 1000*(clock1-clock0), optimizer.param_groups[0]['lr']),end='')
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid,yvalid,  criterion, k)
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
            test_loss, test_acc = test(args, model, device,xtest,ytest,  criterion,k)
            print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc)) 

            mat_list, proto_list_dict, similarities = computeProto(model,xtrain, ytrain,task_id,threshold,proto_list_dict)
            feature_list,proto_index_dict = update_GPM (model, task_id, mat_list, threshold, feature_list, similarities, proto_index_dict)
        
        # save accuracy 
        jj = 0 
        for ii in np.array(task_list)[0:task_id+1]:
            xtest =data[ii]['test']['x']
            ytest =data[ii]['test']['y'] 
            _, acc_matrix[task_id,jj] = test(args, model, device, xtest,ytest,criterion,ii) 
            jj +=1
        print('Accuracies =')
        for i_a in range(task_id+1):
            print('\t',end='')
            for j_a in range(acc_matrix.shape[1]):
                print('{:5.1f}% '.format(acc_matrix[i_a,j_a]),end='')
            print()
        print ('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[task_id].mean())) 
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
    parser = argparse.ArgumentParser(description='Sequential PMNIST with GPM')
    parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=200, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--pc_valid',default=0.05,type=float,
                        help='fraction of training data used for validation')
    parser.add_argument('--num_class', default=100, type=int)
    parser.add_argument('--num_task', default=10, type=int, choices=[5, 10, 20])

    parser.add_argument('--lr', type=float, default=0.04, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr_min', type=float, default=1e-5, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')

    parser.add_argument('--add_description', default='_')
    parser.add_argument('--temp', type=float, default=0.5,
                        help='temperature for loss function')
    parser.add_argument('--supcon', type=float, default=0.0,
                        help='temperature for loss function')
    parser.add_argument('--simi_threshold', type=float, default=0.5)

    args = parser.parse_args()
    args.class_per_task = args.num_class // args.num_task

    log_path = 'checkpoints'
    exp_name = f'CIFA100/{args.num_task}tasks_{args.batch_size_train}batch'
    exp_name += f'_{args.add_description}_T_{args.simi_threshold}_lr_{args.lr}_Supcon_{args.supcon}'
    args.model_path = os.path.join(log_path, exp_name)
    os.makedirs(args.model_path, exist_ok=True)
    file_dir = os.path.join(args.model_path, 'files')
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(file_dir, exist_ok=True)
    file_name = os.path.basename(__file__)
    os.system(f'cp -r {file_name} {file_dir}')
    from logger import FileOutputDuplicator
    sys.stdout = FileOutputDuplicator(sys.stdout, os.path.join(args.model_path, 'log.txt'), 'w')

    print('='*100)
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('='*100)

    main(args)