import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,4,7"
import sys

import torch
import torch.nn as nn
import numpy as np
from skimage.util import montage
from trixi.experiment.pytorchexperiment import PytorchExperiment
from trixi.util import Config
from skimage.transform import resize
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import subprocess
from multiprocessing.pool import Pool
class _UNet3D(PytorchExperiment):
    def setup(self):

        from UNet3D import UNet3D
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet3D(num_classes=4, in_channels=4)
        self.model = torch.nn.DataParallel(self.model) # 多卡并行
        self.load_checkpoint(name="checkpoint/checkpoint_last.pth", path="/home/wwh/wyt/master/20220406-184111_Basic_Unet", save_types=("model",))
        self.model.to("cuda")
        print("setupdone")

class _UNet3DPlus(PytorchExperiment):
    def setup(self):

        
        from UNet3Dplus import UNet3DPlus
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet3DPlus(num_classes=4, in_channels=4)
        self.model = torch.nn.DataParallel(self.model) # 多卡并行
        self.load_checkpoint(name="checkpoint/checkpoint_last.pth", path="/home/wwh/wyt/master/20220411-235337_Basic_Unet", save_types=("model",))
        self.model.to("cuda")
        print("setupdone")


def get_config():
    # Set your own path, if needed.
    data_root_dir = '/home/wwh/wyt/master/BraTS2020_TrainingData/'  # The path where the downloaded dataset is stored.

    c = Config(
        update_from_argv=True,  # If set 'True', it allows to update each configuration by a cmd/terminal parameter.

        # Train parameters
        # num_classes=3,
        num_classes=4,
        in_channels=4,
        # batch_size=8,
        batch_size=4,
        patch_size=64,
        n_epochs=20,
        learning_rate=0.0002,
        fold=0,  # The 'splits.pkl' may contain multiple folds. Here we choose which one we want to use.

        device="cuda",  # 'cuda' is the default CUDA device, you can use also 'cpu'. For more information, see https://pytorch.org/docs/stable/notes/cuda.html

        # Logging parameters
        name='Basic_Unet',
        author='tinawytt',  # Author of this project
        plot_freq=10,  # How often should stuff be shown in visdom
        append_rnd_string=False,  # Appends a random string to the experiment name to make it unique.
        start_visdom=True,  # You can either start a visom server manually or have trixi start it for you.

        do_instancenorm=True,  # Defines whether or not the UNet does a instance normalization in the contracting path
        do_load_checkpoint=False,
        checkpoint_dir='',

        
        base_dir='/home/wwh/wyt/master/',  # Where to log the output of the experiment.

        data_root_dir=data_root_dir,  # The path where the downloaded dataset is stored.
        data_dir=data_root_dir,  # This is where your training and validation data is stored
        data_test_dir=data_root_dir,  # This is where your test data is stored

        split_dir=data_root_dir,  # This is where the 'splits.pkl' file is located, that holds your splits.

        # execute a segmentation process on a specific image using the model
        model_dir=os.path.join('/home/wwh/wyt/master/', ''),  # the model being used for segmentation
    )
    return c

def create_region_from_mask(mask, join_labels: tuple):
    mask_new = np.zeros_like(mask, dtype=np.uint8)
    for l in join_labels:
        mask_new[mask == l] = 1
    return mask_new

def create_region_from_mask2(mask, join_labels: tuple):
    print("enter")
    mask_new = np.zeros_like(mask, dtype=np.uint8)
    for l in join_labels:
        mask_new[mask == l] = 1
    print(np.unique(mask_new))
    return mask_new
# model = UNet3D(num_classes=4, in_channels=4)
# model.load_state_dict(torch.load('/home/wwh/wyt/master/20220317-203046_Basic_Unet/checkpoint/checkpoint_start.pth'))

def onehot_to_0124(tensor):
    permuted_pred=tensor.permute(1,2,3,0)
    pred_tensor_index=torch.topk(permuted_pred,k=1,dim=3)[1]
    pred_tensor_index = torch.squeeze(pred_tensor_index, dim=3)
    pred_tensor_index[pred_tensor_index == 3] = 4
    result_tensor = pred_tensor_index
    return result_tensor


def produce_prediction_and_store_segs(caseid):
    print(caseid)
    # print(model)
    # print(model.model)
    fn_name="/home/wwh/wyt/master/BraTS2020_TrainingData/"+caseid+"/"+caseid+"_preprocessed.npz"
    numpy_array = np.load(fn_name)['data']
    #-------
    # resize
    #-------
    new_shape=(128,128,128)
    result_array=np.zeros((5,128,128,128),dtype=numpy_array.dtype)

    for m in range(0,4):
        result_element = np.zeros(new_shape, dtype=numpy_array.dtype)
        result_element= resize(numpy_array[m].astype(float), new_shape, order=3, clip=True, anti_aliasing=False)
        result_array[m]=result_element

    result_element = np.zeros(new_shape, dtype=numpy_array.dtype)
    unique_labels = np.unique(numpy_array[4])
    for i, c in enumerate(unique_labels):
        mask = numpy_array[4] == c
        reshaped_multihot = resize(mask.astype(float), new_shape, order=1, mode="edge", clip=True, anti_aliasing=False)
        result_element[reshaped_multihot >= 0.5] = c
    result_array[4]=result_element    

    numpy_array=result_array.copy()

    data=torch.from_numpy(numpy_array[0:4]).unsqueeze(0).clone()
    # label=torch.from_numpy(numpy_array[4]).clone()
    label=numpy_array[4].copy()
    #label:-1 -> 0
    label[label==-1]=0
    print("resized")
    data = data.float().to("cuda")
    print(data.shape)
    model.model.eval()
    with torch.no_grad():
        print("enter")
        # pred_tuple=model.model(data)
        # pred_tensor=pred_tuple[0]
        
        pred_tensor = model.model(data)
        print("here")
        # print(pred_tensor.shape)

    squeezed_pred_tensor= pred_tensor.squeeze()#4*128*128*128
    
    result_tensor=onehot_to_0124(squeezed_pred_tensor)
    result_array=result_tensor.cpu().detach().numpy()
    np.savez_compressed(os.path.join('/home/wwh/wyt/master/BraTS2020_TrainingData/'+caseid+'/predict', "%s.npz" % caseid), data=result_array)
    np.savez_compressed(os.path.join('/home/wwh/wyt/master/BraTS2020_TrainingData/'+caseid+'/gt', "%s.npz" % caseid), data=label)
    print("finish "+caseid)


def supv_produce_prediction_and_store_segs(caseid):
    print(caseid)
    # print(model)
    # print(model.model)
    fn_name="/home/wwh/wyt/master/BraTS2020_TrainingData/"+caseid+"/"+caseid+"_preprocessed.npz"
    numpy_array = np.load(fn_name)['data']
    #-------
    # resize
    #-------
    new_shape=(128,128,128)
    result_array=np.zeros((5,128,128,128),dtype=numpy_array.dtype)

    for m in range(0,4):
        result_element = np.zeros(new_shape, dtype=numpy_array.dtype)
        result_element= resize(numpy_array[m].astype(float), new_shape, order=3, clip=True, anti_aliasing=False)
        result_array[m]=result_element

    result_element = np.zeros(new_shape, dtype=numpy_array.dtype)
    unique_labels = np.unique(numpy_array[4])
    for i, c in enumerate(unique_labels):
        mask = numpy_array[4] == c
        reshaped_multihot = resize(mask.astype(float), new_shape, order=1, mode="edge", clip=True, anti_aliasing=False)
        result_element[reshaped_multihot >= 0.5] = c
    result_array[4]=result_element    

    numpy_array=result_array.copy()

    data=torch.from_numpy(numpy_array[0:4]).unsqueeze(0).clone()
    # label=torch.from_numpy(numpy_array[4]).clone()
    label=numpy_array[4].copy()
    #label:-1 -> 0
    label[label==-1]=0
    # print("resized")
    data = data.float().to("cuda")
    # print(data.shape)
    model.model.eval()
    with torch.no_grad():
        # print("enter")
        pred_tuple=model.model(data)
        pred_tensor=pred_tuple[0]
        
        # pred_tensor = model.model(data)
        # print("here")
        # print(pred_tensor.shape)

    squeezed_pred_tensor= pred_tensor.squeeze()#4*128*128*128
    
    result_tensor=onehot_to_0124(squeezed_pred_tensor)
    result_array=result_tensor.cpu().detach().numpy()
    np.savez_compressed(os.path.join('/home/wwh/wyt/master/BraTS2020_TrainingData/'+caseid+'/predict', "%s.npz" % caseid), data=result_array)
    np.savez_compressed(os.path.join('/home/wwh/wyt/master/BraTS2020_TrainingData/'+caseid+'/gt', "%s.npz" % caseid), data=label)
    print("finish "+caseid)


def SUPV_display_result(fn_name="/home/wwh/wyt/master/BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_preprocessed.npz"):
    caseid=fn_name[-20:-17]
    print(caseid)
    numpy_array = np.load(fn_name)['data']
    #-------
    # resize
    #-------
    new_shape=(128,128,128)
    result_array=np.zeros((5,128,128,128),dtype=numpy_array.dtype)

    for m in range(0,4):
        result_element = np.zeros(new_shape, dtype=numpy_array.dtype)
        result_element= resize(numpy_array[m].astype(float), new_shape, order=3, clip=True, anti_aliasing=False)
        result_array[m]=result_element

    result_element = np.zeros(new_shape, dtype=numpy_array.dtype)
    unique_labels = np.unique(numpy_array[4])
    for i, c in enumerate(unique_labels):
        mask = numpy_array[4] == c
        reshaped_multihot = resize(mask.astype(float), new_shape, order=1, mode="edge", clip=True, anti_aliasing=False)
        result_element[reshaped_multihot >= 0.5] = c
    result_array[4]=result_element    

    numpy_array=result_array.copy()

    data=torch.from_numpy(numpy_array[0:4]).unsqueeze(0).clone()
    label=torch.from_numpy(numpy_array[4]).clone()
    # print(np.unique(numpy_array[4]))
    data = data.float().to("cuda")
    model.model.eval()
    with torch.no_grad():
        pred_tuple=model.model(data)
        pred_tensor=pred_tuple[0]
        # pred_tensor = model.model(data)

    squeezed_pred_tensor= pred_tensor.squeeze()#4*128*128*128



    result_tensor=onehot_to_0124(squeezed_pred_tensor)
    result_array=result_tensor.cpu().detach().numpy()
    data_array=data.squeeze().cpu().detach().numpy()

    img_array=data_array[0,64].reshape(1,128,128)
    for c in range(1,4):
        cur_channel_img=data_array[c,64]
        reshaped_cur=cur_channel_img.reshape(1,128,128)
        img_array=np.concatenate((img_array,reshaped_cur),axis=0)
    img_final=montage(img_array,grid_shape=(1,4))

    label_array=label.numpy()
    # print(np.unique(label_array))
    reshaped_label=label_array[64].reshape(1,128,128)
    # print(np.unique(reshaped_label))
    mask_class0=create_region_from_mask(reshaped_label,(0,0))
    mask_class1=create_region_from_mask(reshaped_label,(1,1))
    mask_class2=create_region_from_mask(reshaped_label,(2,2))
    mask_class4=create_region_from_mask(reshaped_label,(4,4))
    mask_all=mask_class0
    mask_all=np.concatenate((mask_all,mask_class1),axis=0)
    mask_all=np.concatenate((mask_all,mask_class2),axis=0)
    mask_all=np.concatenate((mask_all,mask_class4),axis=0)
    mask_label_final=montage(mask_all,grid_shape=(1,4))
    # print(np.unique(result_array))
    reshaped_predict=result_array[64].reshape(1,128,128)
    # print(np.unique(reshaped_predict))
    mask_class0=create_region_from_mask(reshaped_predict,(0,0))
    mask_class1=create_region_from_mask(reshaped_predict,(1,1))
    mask_class2=create_region_from_mask(reshaped_predict,(2,2))
    mask_class4=create_region_from_mask(reshaped_predict,(4,4))
    mask_all=mask_class0
    mask_all=np.concatenate((mask_all,mask_class1),axis=0)
    mask_all=np.concatenate((mask_all,mask_class2),axis=0)
    mask_all=np.concatenate((mask_all,mask_class4),axis=0)
    mask_predict_final=montage(mask_all,grid_shape=(1,4))



    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (128, 512))
    ax1.imshow(img_final, cmap = 'gray')
    ax1.imshow(np.ma.masked_where(mask_label_final == 0, mask_label_final),cmap='autumn', alpha=0.6)
    ax2.imshow(img_final, cmap ='gray')
    ax2.imshow(np.ma.masked_where(mask_predict_final == 0, mask_predict_final),cmap='cool', alpha=0.6)

    plt.savefig(caseid+'.jpg')
    plt.clf()

def display_result(fn_name="/home/wwh/wyt/master/BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_preprocessed.npz"):
    caseid=fn_name[-20:-17]
    print(caseid)
    numpy_array = np.load(fn_name)['data']
    #-------
    # resize
    #-------
    new_shape=(128,128,128)
    result_array=np.zeros((5,128,128,128),dtype=numpy_array.dtype)

    for m in range(0,4):
        result_element = np.zeros(new_shape, dtype=numpy_array.dtype)
        result_element= resize(numpy_array[m].astype(float), new_shape, order=3, clip=True, anti_aliasing=False)
        result_array[m]=result_element

    result_element = np.zeros(new_shape, dtype=numpy_array.dtype)
    unique_labels = np.unique(numpy_array[4])
    for i, c in enumerate(unique_labels):
        mask = numpy_array[4] == c
        reshaped_multihot = resize(mask.astype(float), new_shape, order=1, mode="edge", clip=True, anti_aliasing=False)
        result_element[reshaped_multihot >= 0.5] = c
    result_array[4]=result_element    

    numpy_array=result_array.copy()

    data=torch.from_numpy(numpy_array[0:4]).unsqueeze(0).clone()
    label=torch.from_numpy(numpy_array[4]).clone()
    # print(np.unique(numpy_array[4]))
    data = data.float().to("cuda")
    model.model.eval()
    with torch.no_grad():
        # pred_tuple=model.model(data)
        # pred_tensor=pred_tuple[0]
        pred_tensor = model.model(data)

    squeezed_pred_tensor= pred_tensor.squeeze()#4*128*128*128



    result_tensor=onehot_to_0124(squeezed_pred_tensor)
    result_array=result_tensor.cpu().detach().numpy()
    data_array=data.squeeze().cpu().detach().numpy()

    img_array=data_array[0,64].reshape(1,128,128)
    for c in range(1,4):
        cur_channel_img=data_array[c,64]
        reshaped_cur=cur_channel_img.reshape(1,128,128)
        img_array=np.concatenate((img_array,reshaped_cur),axis=0)
    img_final=montage(img_array,grid_shape=(1,4))

    label_array=label.numpy()
    # print(np.unique(label_array))
    reshaped_label=label_array[64].reshape(1,128,128)
    # print(np.unique(reshaped_label))
    mask_class0=create_region_from_mask(reshaped_label,(0,0))
    mask_class1=create_region_from_mask(reshaped_label,(1,1))
    mask_class2=create_region_from_mask(reshaped_label,(2,2))
    mask_class4=create_region_from_mask(reshaped_label,(4,4))
    mask_all=mask_class0
    mask_all=np.concatenate((mask_all,mask_class1),axis=0)
    mask_all=np.concatenate((mask_all,mask_class2),axis=0)
    mask_all=np.concatenate((mask_all,mask_class4),axis=0)
    mask_label_final=montage(mask_all,grid_shape=(1,4))
    # print(np.unique(result_array))
    reshaped_predict=result_array[64].reshape(1,128,128)
    # print(np.unique(reshaped_predict))
    mask_class0=create_region_from_mask(reshaped_predict,(0,0))
    mask_class1=create_region_from_mask(reshaped_predict,(1,1))
    mask_class2=create_region_from_mask(reshaped_predict,(2,2))
    mask_class4=create_region_from_mask(reshaped_predict,(4,4))
    mask_all=mask_class0
    mask_all=np.concatenate((mask_all,mask_class1),axis=0)
    mask_all=np.concatenate((mask_all,mask_class2),axis=0)
    mask_all=np.concatenate((mask_all,mask_class4),axis=0)
    mask_predict_final=montage(mask_all,grid_shape=(1,4))



    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (128, 512))
    ax1.imshow(img_final, cmap = 'gray')
    ax1.imshow(np.ma.masked_where(mask_label_final == 0, mask_label_final),cmap='autumn', alpha=0.6)
    ax2.imshow(img_final, cmap ='gray')
    ax2.imshow(np.ma.masked_where(mask_predict_final == 0, mask_predict_final),cmap='cool', alpha=0.6)

    plt.savefig(caseid+'.jpg')
    plt.clf()

if __name__ == '__main__':
    
    sys.path.append("/home/wwh/wyt/master/networks/")
    
    c = get_config()
    withoutSUPV=False
    if withoutSUPV:
        model = _UNet3D(config=c, name=c.name, n_epochs=c.n_epochs,seed=42, append_rnd_to_name=c.append_rnd_string, globs=globals())
    else:
        model = _UNet3DPlus(config=c, name=c.name, n_epochs=c.n_epochs,seed=42, append_rnd_to_name=c.append_rnd_string, globs=globals())
    global A
    A=model
    model.setup()
    nlist=['361','325','322','323','315','307','295','291','283','277']
    for n in nlist:
        if withoutSUPV:
            display_result(fn_name="/home/wwh/wyt/master/BraTS2020_TrainingData/BraTS20_Training_"+n+"/BraTS20_Training_"+n+"_preprocessed.npz")
        else:
            SUPV_display_result(fn_name="/home/wwh/wyt/master/BraTS2020_TrainingData/BraTS20_Training_"+n+"/BraTS20_Training_"+n+"_preprocessed.npz")
    
    # run below lines before run region based evaluation
    caseids=[]
    for i in range(1,370):
        caseid="BraTS20_Training_"+str(i//100)+str(i//10%10)+str(i%10)
        subprocess.call("mkdir "+"/home/wwh/wyt/master/BraTS2020_TrainingData/"+caseid+"/predict",shell=True)
        subprocess.call("mkdir "+"/home/wwh/wyt/master/BraTS2020_TrainingData/"+caseid+"/gt",shell=True)
        # subprocess.call("mv BraTS2020_TrainingData/"+caseid+"/predict/* BraTS2020_TrainingData_supv1/"+caseid+"/predict/",shell=True)
        # subprocess.call("mv BraTS2020_TrainingData/"+caseid+"/gt/* BraTS2020_TrainingData_supv1/"+caseid+"/gt/",shell=True)
        caseids.append(caseid)
    
    for case in caseids:
        if withoutSUPV:
            produce_prediction_and_store_segs(case)
        else:
            supv_produce_prediction_and_store_segs(case)
    
    
    