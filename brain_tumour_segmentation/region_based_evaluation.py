from copy import deepcopy
from multiprocessing.pool import Pool

# from medpy import metric
import os
import numpy as np



def get_brats_regions():
    
    regions = {
        "whole tumor": (1, 2, 4),
        "tumor core": (1, 4),
        "enhancing tumor": (4,)
    }
    return regions



def create_region_from_mask(mask, join_labels: tuple):
    mask_new = np.zeros_like(mask, dtype=np.uint8)
    for l in join_labels:
        mask_new[mask == l] = 1
    return mask_new

def dicecoefficient(result, reference):
    """
    Dice coefficient
    
    Computes the Dice coefficient (also known as Sorensen index) between the binary
    objects in two images.
    
    The metric is defined as
    
    .. math::
        
        DC=\frac{2|A\cap B|}{|A|+|B|}
        
    , where :math:`A` is the first and :math:`B` the second set of samples (here: binary objects).
    
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    
    Returns
    -------
    dc : float
        The Dice coefficient between the object(s) in ```result``` and the
        object(s) in ```reference```. It ranges from 0 (no overlap) to 1 (perfect overlap).
        
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    # inter=np.bitwise_xor(result,reference)#我改的np.bitwise_xor
    intersection = np.count_nonzero(result & reference)#&每个像素点做and运算，会不会太严格了
    
    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)
    epsilon=1e-7
    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc=0.0
        # print("ZeroDivisionError")
        # dc =1.0 /1.0+epsilon*float(size_i1)
        # print(dc)
    return dc

def evaluate_case(file_pred: str, file_gt: str, regions):
    image_gt = np.load(file_gt)['data']
    image_pred = np.load(file_pred)['data']
    results = []
    for r in regions:
        mask_pred = create_region_from_mask(image_pred, r)
        mask_gt = create_region_from_mask(image_gt, r)
        dc = np.nan if np.sum(mask_gt) == 0 or np.sum(mask_pred) == 0 else dicecoefficient(mask_pred, mask_gt)
        results.append(dc)
    return results

def getFileList(type=""):
    rList=[]
    train_list=['002', '003', '005', '006', '008', '009', '011', '013', '015', '017', '019', '021', '022', '023', '024', '025', '026', '027', '028', '032', '034', '036', '037', '038', '039', '040', '041', '042', '043', '044', '046', '048', '049', '050', '052', '053', '055', '056', '058', '059', '060', '061', '063', '064', '065', '067', '071', '073', '074', '075', '077', '078', '079', '083', '084', '086', '087', '088', '089', '091', '092', '094', '095', '096', '097', '098', '100', '107', '108', '110', '111', '112', '114', '117', '119', '120', '121', '122', '123', '125', '127', '129', '130', '131', '133', '134', '135', '137', '138', '140', '143', '145', '146', '147', '148', '149', '153', '155', '157', '160', '161', '164', '166', '168', '170', '172', '173', '175', '179', '180', '182', '183', '184', '186', '187', '188', '190', '191', '192', '194', '196', '197', '198', '199', '201', '203', '204', '205', '206', '208', '211', '212', '213', '214', '215', '216', '218', '220', '222', '223', '224', '225', '228', '229', '230', '232', '233', '234', '235', '236', '237', '238', '241', '242', '243', '244', '246', '247', '249', '250', '251', '252', '256', '259', '260', '264', '265', '266', '269', '271', '272', '273', '274', '276', '280', '283', '285', '289', '290', '292', '293', '294', '295', '296', '297', '300', '301', '302', '303', '305', '307', '308', '310', '314', '317', '318', '319', '320', '322', '326', '330', '332', '333', '334', '335', '336', '338', '339', '342', '344', '347', '350', '351', '352', '353', '355', '356', '358', '360', '363', '366', '367', '369']
    val_list=['004', '010', '012', '016', '018', '029', '030', '031', '045', '051', '066', '072', '076', '080', '082', '090', '093', '101', '103', '106', '113', '124', '126', '128', '141', '142', '144', '150', '151', '158', '163', '171', '178', '181', '185', '189', '193', '200', '202', '209', '219', '221', '231', '245', '248', '253', '257', '258', '263', '267', '275', '281', '282', '286', '287', '304', '309', '311', '312', '315', '321', '323', '324', '325', '328', '329', '331', '343', '354', '357', '359', '364', '368']
    test_list=['174', '349', '337', '156', '362', '239', '118', '261', '054', '298', '348', '345', '115', '177', '081', '014', '227', '278', '159', '195', '136', '109', '033', '165', '255', '132', '217', '277', '299', '279', '104', '001', '284', '240', '007', '313', '102', '152', '226', '167', '340', '162', '139', '057', '361', '047', '020', '270', '105', '154', '316', '288', '169', '068', '116', '268', '207', '069', '262', '327', '306', '035', '062', '365', '254', '210', '176', '291', '099', '085', '070', '341', '346']
    for i in range(1,370):
        id=str(i//100)+str(i//10%10)+str(i%10)
        caseid="BraTS20_Training_"+id
        path=caseid+"/"+type+"/"+caseid+".npz"
        # if(id in val_list):
        #     rList.append(path)
        rList.append(path)
        path=""
    return rList

def evaluate_regions(folder: str, regions: dict, processes=2):
    region_names = list(regions.keys())
    files_in_pred = getFileList(type="predict")
    files_in_gt = getFileList(type="gt")
    
    assert len(files_in_pred) == len(files_in_gt), "lack of some gt or pred"
    
    files_in_gt.sort()
    files_in_pred.sort()

    # run for all cases
    full_filenames_gt = [os.path.join(folder, i) for i in files_in_gt]
    full_filenames_pred = [os.path.join(folder, i) for i in files_in_pred]

    p = Pool(processes)
    res = p.starmap(evaluate_case, zip(full_filenames_pred, full_filenames_gt, [list(regions.values())] * len(files_in_gt)))
    p.close()
    p.join()

    all_results = {r: [] for r in region_names}
    with open(os.path.join(folder, 'summary.csv'), 'w') as f:
        f.write("casename")
        for r in region_names:
            f.write(",%s" % r)
        f.write("\n")
        for i in range(len(files_in_pred)):
            f.write(files_in_pred[i][:-33])
            result_here = res[i]
            for k, r in enumerate(region_names):
                dc = result_here[k]
                f.write(",%02.4f" % dc)
                all_results[r].append(dc)
            f.write("\n")

        f.write('mean')
        for r in region_names:
            f.write(",%02.4f" % np.nanmean(all_results[r]))
        f.write("\n")
        f.write('median')
        for r in region_names:
            f.write(",%02.4f" % np.nanmedian(all_results[r]))
        f.write("\n")

        f.write('mean (nan is 1)')
        for r in region_names:
            tmp = np.array(all_results[r])
            tmp[np.isnan(tmp)] = 1
            f.write(",%02.4f" % np.mean(tmp))
        f.write("\n")
        f.write('median (nan is 1)')
        for r in region_names:
            tmp = np.array(all_results[r])
            tmp[np.isnan(tmp)] = 1
            f.write(",%02.4f" % np.median(tmp))
        f.write("\n")

if __name__ == '__main__':
    
    
    evaluate_regions(folder="/home/wwh/wyt/master/BraTS2020_TrainingData/",regions=get_brats_regions())
