import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
import os.path
from collections import OrderedDict
from PIL import Image
import numpy as np
import random
import argparse,time
from copy import deepcopy
from dataloader import cifar100_superclass as data_loader
import augmentations
augmentations.IMAGE_SIZE = 32

## Define LeNet model 
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class LeNet(nn.Module):
    def __init__(self,taskcla):
        super(LeNet, self).__init__()
        self.act=OrderedDict()
        self.map =[]
        self.ksize=[]
        self.in_channel =[]
        
        self.map.append(32)
        self.conv1 = nn.Conv2d(3, 20, 5, bias=False, padding=2)

        s=compute_conv_output_size(32,5,1,2)
        s=compute_conv_output_size(s,3,2,1)
        self.ksize.append(5)
        self.in_channel.append(3)        
        self.map.append(s)
        self.conv2 = nn.Conv2d(20, 50, 5, bias=False, padding=2)
        
        s=compute_conv_output_size(s,5,1,2)
        s=compute_conv_output_size(s,3,2,1)
        self.ksize.append(5)
        self.in_channel.append(20)        
        self.smid=s
        self.map.append(50*self.smid*self.smid)
        self.maxpool=torch.nn.MaxPool2d(3,2,padding=1)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0)
        self.drop2=torch.nn.Dropout(0)
        self.lrn = torch.nn.LocalResponseNorm(4,0.001/9.0,0.75,1)

        self.fc1 = nn.Linear(50*self.smid*self.smid,800, bias=False)
        self.fc2 = nn.Linear(800,500, bias=False)
        self.map.extend([800])
        
        self.taskcla = taskcla
        self.fc3=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.fc3.append(torch.nn.Linear(500,n,bias=False))
        
    def forward(self, x):
        bsz = deepcopy(x.size(0))
        self.act['conv1']=x
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.lrn(self.relu(x))))

        self.act['conv2']=x
        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.lrn (self.relu(x))))

        x=x.reshape(bsz,-1)
        self.act['fc1']=x
        x = self.fc1(x)
        x = self.drop2(self.relu(x))

        self.act['fc2']=x        
        x = self.fc2(x)
        x = self.drop2(self.relu(x))
        feature = F.normalize(x, dim=1)

        y=[]
        for t,i in self.taskcla:
            y.append(self.fc3[t](x))
            
        return y,feature

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

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

def train(args, model, device, x,y, optimizer,criterion, task_id, crop, to_tensor):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)

    train_transform = transforms.Compose([
        transforms.ToTensor(),     
    ])
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        bsz = target.shape[0]
        optimizer.zero_grad()  

        output,feature = model(data)
        loss = 0.0
        if args.supcon > 0.0:
            tmp_img = data.clone()
            tmp_aug = data.clone()
            for index,img in enumerate(data):     
                img = Image.fromarray(np.uint8(img.permute(1,2,0).cpu().numpy()*255))   
            
                img_ori = train_transform(img)
                tmp_img[index] = img_ori

                img_aug = crop(img)
                img_aug = augmentations.aug(img_aug,to_tensor)
                tmp_aug[index] = img_aug 
            imgs_ori = tmp_img
            imgs_aug = tmp_aug

            shuffle_idx = torch.randperm(bsz)
            mapping = {k:v for (v,k) in enumerate(shuffle_idx)}
            reverse_idx = torch.LongTensor([mapping[k] for k in sorted(mapping.keys())])     
            output_aug,feature_aug = model(imgs_aug[shuffle_idx])
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

def train_projected(args,model,device,x,y,optimizer,criterion,feature_mat,task_id, crop, to_tensor):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        bsz = target.shape[0]
        optimizer.zero_grad()   

        output,feature = model(data)
        loss = 0.0
        if args.supcon > 0.0:
            tmp_aug = data.clone()
            for index,img in enumerate(data):     
                img_aug = Image.fromarray(np.uint8(img.permute(1,2,0).cpu().numpy()*255))      
                img_aug = crop(img_aug)
                img_aug = augmentations.aug(img_aug,to_tensor)
                tmp_aug[index] = img_aug 
            img_aug = tmp_aug

            shuffle_idx = torch.randperm(bsz)
            mapping = {k:v for (v,k) in enumerate(shuffle_idx)}
            reverse_idx = torch.LongTensor([mapping[k] for k in sorted(mapping.keys())])     
            output_aug,feature_aug = model(img_aug[shuffle_idx])
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
        kk = 0 
        for k, (m,params) in enumerate(model.named_parameters()):
            if k<4 and len(params.size())!=1:
                sz =  params.grad.data.size(0)
                feature_class_mat = feature_mat[kk]
                for i in range(len(feature_class_mat)):
                    params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                            feature_class_mat[i]).view(params.size())
                kk +=1
            elif (k<4 and len(params.size())==1) and task_id !=0 :
                params.grad.data.fill_(0)

        optimizer.step()

def test(args, model, device, x, y, criterion, task_id):
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
            proto_class_dict = {}
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
                        U,S,Vh = np.linalg.svd(activation_task, full_matrices=False)
                        sval_total = (S**2).sum()
                        sval_ratio = (S**2)/sval_total
                        r = np.sum(np.cumsum(sval_ratio)<threshold[i])
                        feature_class_list.append(U[:,0:r])
                        proto_class_index_dict[j] = len(feature_class_list)-1
                else:
                    U,S,Vh = np.linalg.svd(activation_task, full_matrices=False)
                    sval_total = (S**2).sum()
                    sval_ratio = (S**2)/sval_total
                    r = np.sum(np.cumsum(sval_ratio)<threshold[i])
                    feature_class_list.append(U[:,0:r])
                    proto_class_index_dict[j] = j
            
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
                        
                        sval_total = (S**2).sum()
                        sval_ratio = (S**2)/sval_total
                        r = np.sum(np.cumsum(sval_ratio)<threshold[i])
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
                    U,S,Vh = np.linalg.svd(activation_task, full_matrices=False)
                    # proto_class_dict[j] = j
                    sval_total = (S**2).sum()
                    sval_ratio = (S**2)/sval_total
                    r = np.sum(np.cumsum(sval_ratio)<threshold[i])
                    feature_class_list.append(U[:,0:r])
                    proto_class_index_dict[pre_class_count+task_id] = len(feature_class_list)-1
    print('-'*40)
    print('Gradient Constraints Summary')
    print('-'*40)
    for i in range(len(feature_list)):
        print ('Layer {} : {}/{}'.format(i+1,feature_list[i][0].shape[1], feature_list[i][0].shape[0]))
    print('-'*40)
    return feature_list,proto_index_dict

def computeProto(model, xtrain, ytrain, task_id, threshold, proto_list_dict = {}):
    r=np.arange(xtrain.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).cuda()

    batch_list=[2*12,100,100,125,125] 
    pad = 2
    p1d = (2, 2, 2, 2)
    act_keys = list(model.act.keys())
    mat_list=[]
    similarity_all = []
    features_all = {}
    labels = np.array([])
    model.eval()
    with torch.no_grad():
        for j in range(0,len(r),args.batch_size_train):
            if j+args.batch_size_train<=len(r): b=r[j:j+args.batch_size_train]
            else: b=r[j:]
            data = xtrain[b]
            if data.shape[0] != args.batch_size_train:
                continue
            data, target = data.cuda(), ytrain[b].cuda()
            output,_ = model(data)
            pred = output[task_id].argmax(dim=1, keepdim=True) 
            index,_ = torch.where(pred == target.view_as(pred))

            for i in range(len(model.act.keys())):
                act_key = act_keys[i]
                if i in features_all.keys():
                    features_all[i] = np.concatenate((features_all[i],model.act[act_key][index].detach().cpu().numpy()))
                else:
                    features_all[i] = model.act[act_key][index].detach().cpu().numpy()
            labels = np.hstack((labels,target[index].cpu().numpy()))
    labels_set = np.unique(labels)
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
            index = np.where(item == labels)[0]
            feature_classwise = features[index]
            bsz = min(bsz,feature_classwise.shape[0])
            proto = np.mean(feature_classwise, axis=0)

            if task_id == 0 and item == 0:
                similarities.append(-1)
            else:
                similarity = compute_similarity_with_proto(model,proto,proto_class_list)
                similarities.append(similarity)
            proto_class_list.append(proto)

            if i<2:
                ksz= model.ksize[i]
                s=compute_conv_output_size(model.map[i],model.ksize[i],1,pad)
                mat = np.zeros((model.ksize[i]*model.ksize[i]*model.in_channel[i],s*s*bsz))
                act = F.pad(torch.from_numpy(feature_classwise), p1d, "constant", 0).detach().cpu().numpy()
            
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
    ## Device Setting 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_torch(args.seed)

    # Choose any task order - ref {yoon et al. ICLR 2020}
    task_order = [np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
                  np.array([15, 12, 5, 9, 7, 16, 18, 17, 1, 0, 3, 8, 11, 14, 10, 6, 2, 4, 13, 19]),
                  np.array([17, 1, 19, 18, 12, 7, 6, 0, 11, 15, 10, 5, 13, 3, 9, 16, 4, 14, 2, 8]),
                  np.array([11, 9, 6, 5, 12, 4, 0, 10, 13, 7, 14, 3, 15, 16, 8, 1, 2, 19, 18, 17]),
                  np.array([6, 14, 0, 11, 12, 17, 13, 4, 9, 1, 7, 19, 8, 10, 3, 15, 18, 5, 2, 16])]


    data, taskcla = data_loader.cifar100_superclass_python(task_order[args.t_order], group=5, validation=True)
    test_data,_   = data_loader.cifar100_superclass_python(task_order[args.t_order], group=5)
    print (taskcla)

    crop = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),                       
        ])
    to_tensor = transforms.Compose([                
            transforms.ToTensor(),     
    ]) 

    acc_matrix=np.zeros((20,20))
    criterion = torch.nn.CrossEntropyLoss()

    task_id = 0

    proto_list_dict = {}
    similarities = None
    for k,ncla in taskcla:
        threshold = np.array([0.98] * 5) + task_id*np.array([0.001] * 5)
     
        print('*'*100)
        print('Task {:2d} ({:s})'.format(k,data[k]['name']))
        print('*'*100)
        xtrain=data[k]['train']['x']
        ytrain=data[k]['train']['y']
        xvalid=data[k]['valid']['x']
        yvalid=data[k]['valid']['y']
        xtest =test_data[k]['test']['x']
        ytest =test_data[k]['test']['y']

        lr = args.lr 
        best_loss=np.inf
        print ('-'*40)
        print ('Task ID :{} | Learning Rate : {}'.format(task_id, lr))
        print ('-'*40)
        
        if task_id==0:
            model = LeNet(taskcla).to(device)
            model.simi_threshold = args.simi_threshold
            print ('Model parameters ---')
            for k_t, (m, param) in enumerate(model.named_parameters()):
                print (k_t,m,param.shape)
            print ('-'*40)
            model.apply(init_weights)
            best_model=get_model(model)
            feature_list =[]
            optimizer = optim.SGD(model.parameters(), lr=lr)

            for epoch in range(1, args.n_epochs+1):
                # Train
                clock0=time.time()
                train(args, model, device, xtrain, ytrain, optimizer, criterion, k, crop, to_tensor)
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
            mat_list, proto_list_dict, similarities = computeProto(model,xtrain,ytrain,task_id,threshold,proto_list_dict)
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
                train_projected(args, model,device,xtrain, ytrain,optimizer,criterion,feature_mat,k,crop, to_tensor)
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
            mat_list, proto_list_dict, similarities = computeProto(model,xtrain,ytrain,task_id,threshold,proto_list_dict)
            feature_list,proto_index_dict = update_GPM (model, task_id, mat_list, threshold, feature_list, similarities, proto_index_dict)
        
        # save accuracy 
        jj = 0 
        for ii in task_order[args.t_order][0:task_id+1]:
            xtest =test_data[ii]['test']['x']
            ytest =test_data[ii]['test']['y']
            _, acc_matrix[task_id,jj] = test(args, model, device, xtest, ytest,criterion,ii) 
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
    print ('Task Order : {}'.format(task_order[args.t_order]))
    print ('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[-1].mean())) 
    bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1]) 
    print ('Backward transfer: {:5.2f}%'.format(bwt))
    print('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000))
    print('-'*50)


if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='Sequential PMNIST with GPM')
    parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=50, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--pc_valid',default=0.05,type=float,
                        help='fraction of training data used for validation')
    parser.add_argument('--num_task', default=20, type=int, choices=[6, 11, 10])
    parser.add_argument('--t_order', type=int, default=0, metavar='TOD',
                        help='random seed (default: 0)')
    parser.add_argument('--num_class', default=100, type=int)

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.03, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr_min', type=float, default=1e-5, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')
    parser.add_argument('--add_description', default='')
    parser.add_argument('--temp', type=float, default=0.5,
                        help='temperature for loss function')
    parser.add_argument('--supcon', type=float, default=0.1,
                        help='temperature for loss function')
    parser.add_argument('--simi_threshold', type=float, default=0.7)

    args = parser.parse_args()
    args.class_per_task = int(args.num_class // args.num_task)

    import sys
    log_path = 'checkpoints'
    exp_name = f'CIFA100_S/20tasks_{args.batch_size_train}batch'
    exp_name += f'_{args.add_description}_T_{args.simi_threshold}_Supcon_{args.supcon}' if args.add_description else ''
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



