

import h5py
import torch
import tables
import numpy as np
import pandas as pd 
from torch import nn
import SimpleITK as sitk 
from progressbar import *
from resnext import resnext50
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from resnet import ResNetParallel
from resnext import ResNeXtParallel
from densenet import DenseNetParallel
from torchvision import transforms
from torch.autograd import Variable
from pytorchtools import EarlyStopping
from sklearn.metrics import roc_auc_score 
from torch.utils.data import Dataset, DataLoader



def getProgressbar(message,size):
    widgets = [message, Percentage(), ' ', Bar(marker='-',left='[',right=']'),
            ' ', ETA()] #see docs for other options

    pbar = ProgressBar(widgets=widgets, maxval=size)

    return pbar

def visualizeImagesTorch(data_train,samp):
    for i in range(3):     
        timg,lb,name = data_train.__getitem__(i + samp)        
        timg = np.asarray(timg)
                            
        print(lb)
        print(name)
        print(timg[0].min())
        print(timg[0].max())

        plt.subplot(131)
        plt.imshow(timg[1,70,:,:],cmap = 'gray')
        plt.subplot(132)
        plt.imshow(timg[0,70,:,:],cmap = 'gray')

        plt.show()
        
class ProstateDatasetHDF5(Dataset):

    def __init__(self, fname,pfname,transforms = None):
        self.fname=fname
        self.file = tables.open_file(fname)
        self.tables = self.file.root
        self.nitems=self.tables.data.shape[0]
        self.pvalueimg = sitk.ReadImage(pfname)
        self.pvalue = sitk.GetArrayFromImage(self.pvalueimg)

        self.file.close()
        self.data = None
        self.mask = None 
        self.sdf = None
        self.rf = None
        self.names = None
        self.labels = None 
        self.transforms = transforms
         
    def __getitem__(self, index):
                
        self.file = tables.open_file(self.fname)
        self.tables = self.file.root
        self.data = self.tables.data
        self.mask = self.tables.mask
        self.labels = self.tables.labels
        pvalue = self.pvalue 
        if "names" in self.tables:
            self.names = self.tables.names

        img = self.data[index,:,:,:]
        mask = self.mask[index,:,:,:]


        if self.names is not None:
            name = self.names[index]
        label = self.labels[index]
        self.file.close() 


        out = np.vstack((img[None],mask[None],pvalue[None]))

        # out = np.vstack((img[None],mask[None],pvalue[None]))
        # out = img[None]
        
        out = torch.from_numpy(out)
        return out,label,name

    def __len__(self):
        return self.nitems


def getData(ppath,dataset, batch_size, num_workers,cv):

    trainfilename = fr"<path to train hdf5 file>"
    valfilename =fr"<path to val hdf5 file>"
    testfilename = fr"<path to test hdf5 file>"

    train = h5py.File(trainfilename,libver='latest',mode='r')
    val = h5py.File(valfilename,libver='latest',mode='r')
    test = h5py.File(testfilename,libver='latest',mode='r')

    trainlabels = np.array(train["labels"])
    vallabels = np.array(val["labels"])
    testlabels = np.array(test["labels"])

    train.close()
    test.close()
    val.close()
    
    zeros = (trainlabels == 1).sum()
    ones = (trainlabels != 1).sum()

    data_train = ProstateDatasetHDF5(trainfilename,ppath)
    data_val = ProstateDatasetHDF5(valfilename,ppath)
    data_test  = ProstateDatasetHDF5(testfilename,ppath)

    # Obtaining the train, val and test dataloader instances and loading them to a dictionary 
    trainLoader = torch.utils.data.DataLoader(dataset=data_train,batch_size = batch_size,num_workers = num_workers,shuffle = True)
    valLoader = torch.utils.data.DataLoader(dataset=data_val,batch_size = batch_size,num_workers = num_workers,shuffle = False) 
    testLoader = torch.utils.data.DataLoader(dataset=data_test,batch_size = batch_size,num_workers = num_workers,shuffle = False) 

    dataLoader = {}
    dataLoader['train'] = trainLoader
    dataLoader['val'] = valLoader
    dataLoader['test'] = testLoader

    return dataLoader, zeros, ones 



def run(mn,device,dataset, shapepath, regionpath, zeros, ones, num_epochs, learning_rate, weightdecay, patience, cv):

    # choosing the architecture 

    model = 
    # model.load_state_dict(torch.load(fr"Data/modelcheckpoints_OLD/miccai50_{cv}_{rtype}/checkpoint.pt"))


    if mn == "resnet":
        model = ResNetParallel(34,shapepath,regionpath)        
    if mn == "densenet":
        model = DenseNetParallel(121,shapepath,regionpath)
    if mn == "resnext":
        model = ResNextParallel(shapepath,regionpath)


    model = nn.DataParallel(model,device_ids=[0,1])
    model.to(f'cuda:{model.device_ids[0]}')

    total = zeros + ones 

    # define weights based on how the training set is balanced
    weights = [zeros/float(total),ones/float(total)]

    class_weights = torch.FloatTensor(weights).cuda(f'cuda:{model.device_ids[0]}')
    criterion=nn.CrossEntropyLoss(weight = class_weights)

    # defining the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weightdecay)

    niter_total=len(dataLoader['train'].dataset)/batch_size

    display = ["val","test"]

    results = {} 
    results["patience"] = patience

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    modelname = fr"parallel/{dataset}_parallel_{mn}"
    parentfolder = r"Data/"

    print(modelname)

    # start training 
    # the predictions and checkpoint will be saved in <parentfolder>/<modelname>
    for epoch in range(num_epochs):

        pred_df_dict = {} 
        results_dict = {} 
        
        
        for phase in ["train","test","val"]:


            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            confusion_matrix=np.zeros((2,2))
            
            loss_vector=[]
            ytrue = [] 
            ypred = [] 
            ynames = [] 
            features = None 


            niter_total_phase=len(dataLoader[phase].dataset)/batch_size
            pbar = getProgressbar(fr'{phase} epoch {epoch}  : ',niter_total_phase)
            pbar.start()

            for ii,(data,label,name) in enumerate(dataLoader[phase]):
                if data.shape[0] != 1:
                    label=label.squeeze().long().to(f'cuda:{model.device_ids[0]}')
                    data = Variable(data.float().cuda(f'cuda:{model.device_ids[0]}'))

                    with torch.set_grad_enabled(phase == 'train'):

                        output,feat = model(data)
                        output = output.squeeze()

                        feat = feat.detach().data.cpu().numpy()
                        features = feat if features is None else np.vstack((features,feat))

                        try:
                            _,pred_label=torch.max(output,1)

                        except:
                            import pdb 
                            pdb.set_trace()

                        probs = F.softmax(output,dim = 1)

                        loss = criterion(probs, label)

                        probs = probs[:,1]

                        loss_vector.append(loss.detach().data.cpu().numpy())

                        if phase=="train":
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()        

                        ypred.extend(probs.cpu().data.numpy().tolist())
                        ytrue.extend(label.cpu().data.numpy().tolist())
                        ynames.extend(list(name))

                        pred_label=pred_label.cpu()
                        label=label.cpu()
                        for p,l in zip(pred_label,label):
                            confusion_matrix[p,l]+=1
                        
                pbar.update(ii)
            pbar.finish()



            total=confusion_matrix.sum()        
            acc=confusion_matrix.trace()/total
            loss_avg=np.mean(loss_vector)
            auc = roc_auc_score(ytrue,ypred)


            columns = ["FileName","True", "Pred","Phase"]

            for fno in range(features.shape[1]):
                columns.append(fr"feat_{fno}")

            
            pred_df = pd.DataFrame(np.column_stack((ynames,ytrue,ypred,[phase]*len(ynames),features)), 
                                columns = columns)

            pred_df_dict[phase] = pred_df

            results_dict[phase] = {} 
            results_dict[phase]["loss"] = loss_avg
            results_dict[phase]["auc"] = auc 
            results_dict[phase]["acc"] = acc 

            if phase == 'train':
                print("Epoch : {}, Phase : {}, Loss : {}, Acc: {}, Auc : {}".format(epoch,phase,loss_avg,acc,auc))
            elif phase in display:
                print("                 Epoch : {}, Phase : {}, Loss : {}, Acc: {}, Auc : {}".format(epoch,phase,loss_avg,acc,auc))
                
            for cl in range(confusion_matrix.shape[0]):
                cl_tp=confusion_matrix[cl,cl]/confusion_matrix[:,cl].sum()

            if phase == 'val':
                df = pred_df_dict["val"].append(pred_df_dict["test"], ignore_index=True)
                early_stopping(loss_avg, model, modelname, df, results_dict,parentfolder =None)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        if early_stopping.early_stop:
            break


if __name__ == "__main__":


    # cross validation splits
    cvs = range(3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # define architecture, resnet, resnext or densenet
    modelnames = ["densenet"]

    # define batch size, number of epochs, learning rate, and weigth decay. 
    batch_size = 4
    num_workers = 8

    num_epochs = 200
    learning_rate = 1e-5
    weightdecay = 1e-3
    
    # patient criteria for early stopping
    # if validation loss increases consecutively for the defined 'patience', network training is stopped
    patience = 10

    # based on either 'infiltrate' regions or 'atlas based shape distention regions' 
    rtype = "together"

    # path to the 
    ppath = "<path to the binary mask of shape different atlas, DA>"

    # loop through the cross validation folds and train the network
    for cv in cvs:
        for mn in modelnames:


            dataset = "<filename of the hdf5 file>"

            # obtain train, val and test dataloader
            dataLoader, zeros, ones = getData(ppath,dataset, batch_size, num_workers,cv)


            shapepath = "<path to the trained checkpoint of M1 corresponding to the cross validation fold>"
            regionpath = "<path to the trained checkpoint of M2 corresponding to the cross validation fold>"

            # Network training 
            run(mn,device,dataset, shapepath, regionpath, zeros, ones, num_epochs, learning_rate, weightdecay, patience, cv)
