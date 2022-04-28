import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
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


from collections import defaultdict
from medpy.io import load
import random
from copy import deepcopy
from scipy.ndimage import map_coordinates, fourier_gaussian
from scipy.ndimage.filters import gaussian_filter, gaussian_gradient_magnitude
from scipy.ndimage.morphology import grey_dilation
from scipy.ndimage.measurements import label as lb
import matplotlib.cm as cm
import SimpleITK as sitk
from collections import OrderedDict
import pickle
from subprocess import check_call


class _UNet3DPlus(PytorchExperiment):
    def setup(self):

        
        from UNet3Dplus import UNet3DPlus
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet3DPlus(num_classes=4, in_channels=4)
        self.model = torch.nn.DataParallel(self.model) # 多卡并行
        self.load_checkpoint(name="checkpoint_last.pth", path="/home/jovyan/main", save_types=("model",))
        self.model.to("cuda")
        print("setupdone")


def get_config():
    # Set your own path, if needed.
    data_root_dir = '/home/jovyan/main/BraTS2020_TrainingData/'  # The path where the downloaded dataset is stored.

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

        
        base_dir='/home/jovyan/main/',  # Where to log the output of the experiment.

        data_root_dir=data_root_dir,  # The path where the downloaded dataset is stored.
        data_dir=data_root_dir,  # This is where your training and validation data is stored
        data_test_dir=data_root_dir,  # This is where your test data is stored

        split_dir=data_root_dir,  # This is where the 'splits.pkl' file is located, that holds your splits.

        # execute a segmentation process on a specific image using the model
        model_dir=os.path.join('/home/jovyan/main/', ''),  # the model being used for segmentation
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


def SUPV_display_result(slice:int,fn_name="/home/wwh/wyt/master/BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_preprocessed.npz"):
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
    # label=torch.from_numpy(numpy_array[4]).clone()
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

    img_array=data_array[0,slice].reshape(1,128,128)
    for c in range(1,4):
        cur_channel_img=data_array[c,slice]
        reshaped_cur=cur_channel_img.reshape(1,128,128)
        img_array=np.concatenate((img_array,reshaped_cur),axis=0)
    img_final=montage(img_array,grid_shape=(1,4))

    # label_array=label.numpy()
    # # print(np.unique(label_array))
    # reshaped_label=label_array[64].reshape(1,128,128)
    # # print(np.unique(reshaped_label))
    # mask_class0=create_region_from_mask(reshaped_label,(0,0))
    # mask_class1=create_region_from_mask(reshaped_label,(1,1))
    # mask_class2=create_region_from_mask(reshaped_label,(2,2))
    # mask_class4=create_region_from_mask(reshaped_label,(4,4))
    # mask_all=mask_class0
    # mask_all=np.concatenate((mask_all,mask_class1),axis=0)
    # mask_all=np.concatenate((mask_all,mask_class2),axis=0)
    # mask_all=np.concatenate((mask_all,mask_class4),axis=0)
    # mask_label_final=montage(mask_all,grid_shape=(1,4))
    # print(np.unique(result_array))
    reshaped_predict=result_array[slice].reshape(1,128,128)
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



    fig, (ax2) = plt.subplots(1, 1, figsize = (128, 512))
    # ax1.imshow(img_final, cmap = 'gray')
    # ax1.imshow(np.ma.masked_where(mask_label_final == 0, mask_label_final),cmap='autumn', alpha=0.6)
    ax2.imshow(img_final, cmap ='gray')
    ax2.imshow(np.ma.masked_where(mask_predict_final == 0, mask_predict_final),cmap='cool', alpha=0.6)

    plt.savefig(caseid+'_'+str(slice)+'.jpg')
    plt.clf()

def visualize_seg(all_data,slice:int):
    
    numpy_array = all_data
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

    img_array=data_array[0,slice].reshape(1,128,128)
    for c in range(1,4):
        cur_channel_img=data_array[c,slice]
        reshaped_cur=cur_channel_img.reshape(1,128,128)
        img_array=np.concatenate((img_array,reshaped_cur),axis=0)
    img_final=montage(img_array,grid_shape=(1,4))

    # label_array=label.numpy()
    # # print(np.unique(label_array))
    # reshaped_label=label_array[64].reshape(1,128,128)
    # # print(np.unique(reshaped_label))
    # mask_class0=create_region_from_mask(reshaped_label,(0,0))
    # mask_class1=create_region_from_mask(reshaped_label,(1,1))
    # mask_class2=create_region_from_mask(reshaped_label,(2,2))
    # mask_class4=create_region_from_mask(reshaped_label,(4,4))
    # mask_all=mask_class0
    # mask_all=np.concatenate((mask_all,mask_class1),axis=0)
    # mask_all=np.concatenate((mask_all,mask_class2),axis=0)
    # mask_all=np.concatenate((mask_all,mask_class4),axis=0)
    # mask_label_final=montage(mask_all,grid_shape=(1,4))
    # print(np.unique(result_array))
    reshaped_predict=result_array[slice].reshape(1,128,128)
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



    fig, (ax2) = plt.subplots(1, 1, figsize = (128, 512))
    # ax1.imshow(img_final, cmap = 'gray')
    # ax1.imshow(np.ma.masked_where(mask_label_final == 0, mask_label_final),cmap='autumn', alpha=0.6)
    ax2.imshow(img_final, cmap ='gray')
    ax2.imshow(np.ma.masked_where(mask_predict_final == 0, mask_predict_final),cmap='cool', alpha=0.6)

    plt.savefig('test'+'_'+str(slice)+'.jpg')
    plt.clf()

def seg_new(fn_name:str):
    SUPV_display_result(fn_name=fn_name,slice=32)
    SUPV_display_result(fn_name=fn_name,slice=64)
    SUPV_display_result(fn_name=fn_name,slice=96)

def subfiles(folder,res, join=True, prefix=None, suffix=None, sort=True):
    # if join:
    #     l = os.path.join
    # else:
    #     l = lambda x, y: y
    # for i in os.listdir(folder):
    #     print(i)
    # res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
    #         and (prefix is None or i.startswith(prefix))
    #         and (suffix is None or i.endswith(suffix))]
    
    dirList=[]
    for i in os.listdir(folder):
        wholepath = os.path.join(folder, i)
        if os.path.isdir(wholepath):
            dirList.append(wholepath)
        if os.path.isfile(wholepath):
            res.append(wholepath)
            if not wholepath.endswith(suffix):
                res.remove(wholepath)
    if dirList:
        for subDir in dirList:
            subfiles(subDir,res,join=False,suffix=".nii.gz")
    if sort:
        res.sort()

def create_nonzero_mask(data):
    from scipy.ndimage import binary_fill_holes
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    # print(4,data.shape[1:]) 155*240*240
    # print(5,data.shape[0]) 4
    for c in range(data.shape[0]):
        this_mask = data[c] != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask


def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]

def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """
    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask, 0)
    # print(9,bbox)

    cropped_data = []
    for c in range(data.shape[0]):
        # if c==0:
        #     print(7,data[0].shape)
        cropped = crop_to_bbox(data[c], bbox)
        # if c==0:
        #     print(8,cropped.shape)
        cropped_data.append(cropped[None])
    data = np.vstack(cropped_data)

    if seg is not None:
        cropped_seg = []
        for c in range(seg.shape[0]):
            
            cropped = crop_to_bbox(seg[c], bbox)
            cropped_seg.append(cropped[None])
        seg = np.vstack(cropped_seg)

    nonzero_mask = crop_to_bbox(nonzero_mask, bbox)[None]
    if seg is not None:
        seg[(seg == 0) & (nonzero_mask == 0)] = nonzero_label
    
    return data, seg, bbox

def crop_data(root_dir='/home/jovyan/main/test/', y_shape=64, z_shape=64):
    # image_dir = os.path.join(root_dir, 'imagesTr')
    image_dir = root_dir
    
    classes = 4

    
    class_stats = defaultdict(int)
    total = 0
    nii_files=[]
    subfiles(image_dir,nii_files, suffix=".nii.gz", join=False)

    # for i in range(0, len(nii_files)):
    #     if nii_files[i].startswith("._"):
    #         nii_files[i] = nii_files[i][2:]
    # print("--------")
    seg_files=[]
    data_files=[]
    data_itk = []
    seg_itk=[]
    count=0
    for f in nii_files:
        count=count+1
        image,metadata=load(f)
        label=metadata.get_sitkimage()
        if "seg" in f:
            seg_files.append(f)
            
            seg_itk.append(label)
            
        else:
            data_itk.append(label)
            data_files.append(f)
        if count==5:
            print(seg_files,data_files)
            data_npy = np.vstack([sitk.GetArrayFromImage(d)[None] for d in data_itk])
            print(1,np.array(data_itk[0].GetSize()))
            print(2,np.array(data_itk[0].GetSize())[[2, 1, 0]])
            
            seg_npy = np.vstack([sitk.GetArrayFromImage(s)[None] for s in seg_itk])
            data_npy= data_npy.astype(np.float32)
            seg_npy= seg_npy.astype(np.float32)
            # npImage = sitk.GetArrayFromImage(label)
            # print(2,npImage)
            # z = int(label.GetSize()[2]/2)
            # plt.figure(figsize=(5,5))
            # plt.imshow(image[:,:,z], 'gray')
            # plt.show()
            properties = OrderedDict()
            properties["original_size_of_raw_data"] = np.array(data_itk[0].GetSize())[[2, 1, 0]]
            properties["original_spacing"] = np.array(data_itk[0].GetSpacing())[[2, 1, 0]]
            properties["list_of_data_files"] = data_files
            properties["seg_file"] = seg_files
            properties["itk_origin"] = data_itk[0].GetOrigin()
            properties["itk_spacing"] = data_itk[0].GetSpacing()
            properties["itk_direction"] = data_itk[0].GetDirection()
            print(4,data_npy.shape,seg_npy.shape)
            data_npy,seg_npy,bbox=crop_to_nonzero(data_npy, seg_npy, nonzero_label=-1)    
            print(5,data_npy.shape,seg_npy.shape)
            properties["crop_bbox"] = bbox
            properties['classes'] = np.unique(seg_npy)
            seg_npy[seg_npy < -1] = 0
            properties["size_after_cropping"] = data_npy[0].shape 
            print(6,properties['classes'],properties['classes'].shape)
            case_id=seg_files[0].split("/")[-1].split(".nii.gz")[0][0:-4]
            all_data = np.vstack((data_npy, seg_npy))
            # np.savez_compressed(os.path.join('/home/jovyan/main/BraTS2020_TrainingData/'+case_id, "%s.npz" % case_id), data=all_data)
            # with open(os.path.join('/home/jovyan/main/BraTS2020_TrainingData/'+case_id, "%s.pkl" % case_id), 'wb') as file:
            #     pickle.dump(properties, file)
            
            count=0
            data_files=[]
            seg_files=[]
            data_itk = []
            seg_itk=[]
    return all_data

def intensity_normalization(all_data):
    # zero-mean normalization
    
    data = all_data[:-1].astype(np.float32)
    seg = all_data[-1:]
    # with open(os.path.join('/home/jovyan/main/test/'+case_identifier, "%s.pkl" % case_identifier), 'rb') as f:
    #     properties = pickle.load(f)
    # print(len(data))
    # print(type(data))
    intensity_properties={}
    intensity=[]
    for i in range(0,len(data)):
        # print(data[i].shape)
        mn=np.mean(data[i])
        std=np.std(data[i])
        lower_bound=np.percentile(data[i],1)
        upper_bound=np.percentile(data[i],99)
        data[i]=np.clip(data[i], lower_bound, upper_bound)
        data[i] = (data[i] - mn) / std
    all_data = np.vstack((data, seg)).astype(np.float32)
    # num_samples = 10000
    # min_percent_coverage = 0.01 # at least 1% of the class voxels need to be selected, otherwise it may be too sparse
    # rndst = np.random.RandomState(1234)
    # class_locs = {}
    # all_classes=[0,1,2,4]
    # for c in all_classes:
    #     all_locs = np.argwhere(all_data[-1] == c)
    #     if len(all_locs) == 0:
    #         class_locs[c] = []
    #         continue
    #     target_num_samples = min(num_samples, len(all_locs))
    #     target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))
    #     selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
    #     class_locs[c] = selected
    #     # print(c, target_num_samples)
    # properties['class_locations'] = class_locs
    # np.savez_compressed(os.path.join('/home/jovyan/main/test/', "test_preprocessed.npz"), data=all_data)
    # np.savez_compressed(os.path.join('/home/jovyan/main/BraTS2020_TrainingData/'+case_identifier, "%s_normalized.npz" % case_identifier), data=all_data)
    # with open(os.path.join('/home/jovyan/main/BraTS2020_TrainingData/'+case_identifier, "%s_preprocessed.pkl" % case_identifier), 'wb') as file:
    #     pickle.dump(properties, file)
    return all_data
    
def gamma_augmentation(all_data,gamma_range=(0.5,2),epsilon=1e-7, per_channel=False,retain_stats = False):
    all_data = all_data
    
    data = all_data[:-1].astype(np.float32)
    seg = all_data[-1:]
    for c in range(len(data)):
        # print(data[c].shape)
        # print(data[c].shape[0])
        if per_channel:
            for channel in range(data[c].shape[0]):
                
                if retain_stats:
                    mn = data[c][channel].mean()
                    sd = data[c][channel].std()
                if np.random.random() < 0.5 and gamma_range[0] < 1:
                    gamma = np.random.uniform(gamma_range[0], 1)
                else:
                    gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
                minm = data[c][channel].min()
                rnge = data[c][channel].max() - minm
                data[c][channel] = np.power(((data[c][channel] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
                if retain_stats:
                    data[c][channel] = data[c][channel] - data[c][channel].mean()
                    data[c][channel] = data[c][channel] / (data[c][channel].std() + 1e-8) * sd
                    data[c][channel] = data[c][channel] + mn
            # all_channel_mn=0
            # all_channel_sd=1
            # # print('before',data[c].mean(),data[c].std())
            # data[c]=data[c]-data[c].mean()
            # data[c]=data[c]/(data[c].std()+1e-8)*all_channel_sd
            # data[c]=data[c]+all_channel_mn
            # # print('after',data[c].mean(),data[c].std())
        else:
            
            # print('before',data[c].mean(),data[c].std())
            if retain_stats:
                mn = data[c].mean()
                sd = data[c].std()
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data[c].min()
            rnge = data[c].max() - minm
            data[c]= np.power(((data[c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
            if retain_stats:
                data[c] = data[c] - data[c].mean()
                data[c] = data[c]/ (data[c].std() + 1e-8) * sd
                data[c] = data[c] + mn
            # print('after',data[c].mean(),data[c].std())
    all_data=np.vstack((data,seg)).astype(np.float32)
    return all_data
    
# additive brightness augmentation
def brightness_augmentation(all_data,mu:float, sigma:float ,per_channel:bool=True):
    data = all_data[:-1].astype(np.float32)
    seg = all_data[-1:]
    for c in range(len(data)):
        if per_channel:
            for channel in range(data[c].shape[0]):
                if np.random.uniform() <= 0.3:
                    rnd_nb = np.random.normal(mu, sigma)
                    data[c][channel] += rnd_nb
        else:
            rnd_nb = np.random.normal(mu, sigma)
            for channel in range(data[c].shape[0]):
                if np.random.uniform() <= 0.3:
                    data[c][channel]+=rnd_nb
    all_data=np.vstack((data,seg)).astype(np.float32)
    return all_data

if __name__ == '__main__':
    
    sys.path.append("/home/jovyan/main/networks/")
    
    c = get_config()
    model = _UNet3DPlus(config=c, name=c.name, n_epochs=c.n_epochs,seed=42, append_rnd_to_name=c.append_rnd_string, globs=globals())
    global A
    A=model
    model.setup()
    # nlist=['004']
    # for n in nlist:
    #     seg_new(fn_name="/home/jovyan/main/BraTS2020_TrainingData/BraTS20_Training_"+n+"/BraTS20_Training_"+n+"_preprocessed.npz")
    all_data=crop_data()
    all_data2=gamma_augmentation(all_data=all_data)
    all_data3=brightness_augmentation(all_data=all_data2,mu=0.0,sigma=1.0)
    all_data4=intensity_normalization(all_data=all_data3)
    visualize_seg(all_data=all_data4,slice=32)
    visualize_seg(all_data=all_data4,slice=64)
    visualize_seg(all_data=all_data4,slice=96)
    
    
    
    
    