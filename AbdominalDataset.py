# Dataloader for abdominal images
import glob
import numpy as np
import dataloaders.niftiio as nio
import dataloaders.transform_utils as trans
import torch
import os
import torch.utils.data as torch_data
import math
import itertools
from PIL import Image
from dataloaders.augs_TIBA import strong_img_aug
import SimpleITK as sitk
from dataloaders.niftiio import read_nii_bysitk

BASEDIR = 'xxx'#'./data/abdominal/'
LABEL_NAME = ["bg", "liver", "rk", "lk", "spleen"]


def get_normalize_op(modality, fids):# modality:   CT or MR,fids for the fold
    def get_CT_statistics(scan_fids):
        total_val = 0
        n_pix = 0
        for fid in scan_fids:
            in_img = read_nii_bysitk(fid)
            total_val += in_img.sum()
            n_pix += np.prod(in_img.shape)
            del in_img
        meanval = total_val / n_pix

        total_var = 0
        for fid in scan_fids:
            in_img = read_nii_bysitk(fid)
            total_var += np.sum((in_img - meanval) ** 2 )
            del in_img
        var_all = total_var / n_pix

        global_std = var_all ** 0.5

        return meanval, global_std

   
    
    ct_mean, ct_std = get_CT_statistics(fids)

    def CT_normalize(x_in):
        return (x_in - ct_mean) / ct_std

    return CT_normalize #, {'mean': ct_mean, 'std': ct_std}



class AbdominalDataset(torch_data.Dataset):
    def __init__(self,  mode, transforms, base_dir, domains: list,  idx_pct = [0.7, 0.1, 0.2], tile_z_dim = 3, extern_norm_fn = None, opt=None):
    
    
        super(AbdominalDataset, self).__init__()
        self.transforms=transforms
        self.is_train = True if mode == 'train' else False
        self.phase = mode
        self.domains = domains
        self.all_label_names = LABEL_NAME
        self.nclass = len(LABEL_NAME)
        self.tile_z_dim = tile_z_dim
        self._base_dir = base_dir
        self.idx_pct = idx_pct
        self.opt=opt
        self.augseg1=strong_img_aug(num_augs=5,flag_using_random_num=True)
        self.augseg2 = strong_img_aug(num_augs=5, flag_using_random_num=True)

        self.img_pids = {}
        for _domain in self.domains: # load file names
            self.img_pids[_domain] = sorted([ fid.split("_")[-1].split(".nii.gz")[0] for fid in glob.glob(self._base_dir + "/" +  _domain  + "/processed/image_*.nii.gz") ], key = lambda x: int(x))

        self.scan_ids = self.__get_scanids(idx_pct)
        self.info_by_scan = None
        self.sample_list = self.__search_samples(self.scan_ids) # image files names according to self.scan_ids

        self.pid_curr_load = self.scan_ids
       
        if extern_norm_fn is None:
            self.normalize_op = get_normalize_op(self.domains[0], [ itm['img_fid'] for _, itm in self.sample_list[self.domains[0]].items() ])
            print(f'{self.phase}_{self.domains[0]}: Using fold data statistics for normalization')
        else:
            self.normalize_op = extern_norm_fn

        print(f'For {self.phase} on {[_dm for _dm in self.domains]} using scan ids {self.pid_curr_load}')

        # load to memory
        self.actual_dataset = self.__read_dataset()
        self.size = len(self.actual_dataset) # 2D

    def __get_scanids(self, idx_pct):

        tr_trids,tr_valids,tr_teids,te_teids={},{},{},{}

        for _domain in self.domains:
            dset_size   = len(self.img_pids[_domain])
            tr_size     = round(dset_size * idx_pct[0])
            val_size    = math.floor(dset_size * idx_pct[1])
            te_size     = dset_size - tr_size - val_size

            tr_teids[_domain]     = self.img_pids[_domain][: te_size]
            tr_valids[_domain]    = self.img_pids[_domain][te_size: te_size + val_size]
            tr_trids[_domain]     = self.img_pids[_domain][te_size + val_size: ]



            te_teids[_domain] = list(itertools.chain(tr_trids[_domain], tr_teids[_domain], tr_valids[_domain]))

        if self.phase == 'train':
         
            xx = round(len(tr_trids[self.domains[0]]) * self.opt.per)
          
            tr_trids[self.domains[0]] = tr_trids[self.domains[0]][:xx]

            return tr_trids
        elif self.phase == 'trval':
            return tr_valids
        elif self.phase == 'trtest':
            return tr_teids
        elif self.phase == 'test':
            return te_teids
        elif self.phase=='testsup':
            return tr_teids
      

    def __search_samples(self, scan_ids):
        """search for filenames for images and masks
        """
        out_list = {}
        for _domain, id_list in scan_ids.items():
            out_list[_domain] = {}
            for curr_id in id_list:
                curr_dict = {}

                _img_fid = os.path.join(self._base_dir, _domain , 'processed'  ,f'image_{curr_id}.nii.gz')
                _lb_fid  = os.path.join(self._base_dir, _domain , 'processed', f'label_{curr_id}.nii.gz')

                curr_dict["img_fid"] = _img_fid
                curr_dict["lbs_fid"] = _lb_fid
                out_list[_domain][str(curr_id)] = curr_dict

        return out_list


    def __read_dataset(self):
        """
        Read the dataset into memory
        """

        out_list = []
        self.info_by_scan = {} # meta data of each scan
        glb_idx = 0 # global index of a certain slice in a certain scan in entire dataset

        for _domain, _sample_list in self.sample_list.items():
            for scan_id, itm in _sample_list.items():
                if scan_id not in self.pid_curr_load[_domain]:
                    continue

                img, _info = nio.read_nii_bysitk(itm["img_fid"], peel_info = True) # get the meta information out
                self.info_by_scan[_domain + '_' + scan_id] = _info

                img = np.float32(img)
                img = self.normalize_op(img)

                lb = nio.read_nii_bysitk(itm["lbs_fid"])
                lb = np.float32(lb)

                img     = np.transpose(img, (1,2,0))
                lb      = np.transpose(lb, (1,2,0))

                assert img.shape[-1] == lb.shape[-1]

                # now start writing everthing in
                # write the beginning frame
                out_list.append( {"img": img[..., 0: 1],
                               "lb":lb[..., 0: 0 + 1],
                               "is_start": True,
                               "is_end": False,
                               "domain": _domain,
                               "nframe": img.shape[-1],
                               "scan_id": _domain + "_" + scan_id,
                               "z_id":0})
                glb_idx += 1

                for ii in range(1, img.shape[-1] - 1):
                    out_list.append( {"img": img[..., ii: ii + 1],
                               "lb":lb[..., ii: ii + 1],
                               "is_start": False,
                               "is_end": False,
                               "nframe": -1,
                               "domain": _domain,
                               "scan_id":_domain + "_" + scan_id,
                               "z_id": ii
                               })
                    glb_idx += 1

                ii += 1 # last frame, note the is_end flag
                out_list.append( {"img": img[..., ii: ii + 1],
                               "lb":lb[..., ii: ii+ 1],
                               "is_start": False,
                               "is_end": True,
                               "nframe": -1,
                               "domain": _domain,
                               "scan_id":_domain + "_" + scan_id,
                               "z_id": ii
                               })
                glb_idx += 1

        return out_list
    

    def __getitem__(self, index):
        index = index % len(self.actual_dataset)
        curr_dict = self.actual_dataset[index]

        is_start    = curr_dict["is_start"]
        is_end      = curr_dict["is_end"]
        nframe      = np.int32(curr_dict["nframe"])
        scan_id     = curr_dict["scan_id"]
        z_id        = curr_dict["z_id"]

        sample = {"is_start": is_start,"is_end": is_end,"nframe": nframe,"scan_id": scan_id,"z_id": z_id}

        if self.phase!='train':
            img = curr_dict['img']
            lb = curr_dict['lb']
            imgori=curr_dict['img']
          
            img = np.float32(img)
            lb = np.float32(lb)
            imgori= np.float32(imgori)
    
            img = np.transpose(img, (2, 0, 1))
            lb  = np.transpose(lb, (2, 0, 1))
            imgori= np.transpose(imgori, (2, 0, 1))

            img = torch.from_numpy( img )
            lb  = torch.from_numpy( lb )
            imgori= torch.from_numpy( imgori )

            if self.tile_z_dim > 1:
               img = img.repeat( [ self.tile_z_dim, 1, 1] )
               imgori = imgori.repeat( [ self.tile_z_dim, 1, 1] )
        

            sample['img1']=img
            sample['lb']=lb
            sample['ori']=imgori
        
        else:
            if self.opt.aug_type == 'augseg_two_concat':
              
                comp = np.concatenate([curr_dict["img"], curr_dict["lb"]], axis=-1)
                imgori = curr_dict['img']
             
                img, lb = self.transforms(comp, c_img=1, c_label=1, nclass=self.nclass, is_train=self.is_train,use_onehot=False)

                
                img = img.squeeze(2)
                img = convert_to_png(img, img.min(), img.max())
                img = Image.fromarray(img)
                img = img.convert('L')

                img1 = self.augseg1(img)
                img2 = self.augseg2(img)

                img1, img2 = np.array(img1), np.array(img2)
                img1, img2 = np.reshape(img1, (img1.shape[0], img1.shape[1], 1)), np.reshape(img2, (img2.shape[0], img2.shape[1], 1))

                img1, img2 = np.float32(img1), np.float32(img2)
                imgori = np.float32(imgori)
                lb = np.float32(lb)

                img1 /= 127.5
                img1 -= 1.0

                img2 /= 127.5
                img2 -= 1.0

                img1 = np.transpose(img1, (2, 0, 1))
                img2 = np.transpose(img2, (2, 0, 1))
                imgori = np.transpose(imgori, (2, 0, 1))
                lb = np.transpose(lb, (2, 0, 1))

                img1, img2 = torch.from_numpy(img1), torch.from_numpy(img2)
                imgori = torch.from_numpy(imgori)
                lb = torch.from_numpy(lb)

                if self.tile_z_dim > 1:
                    img1, img2 = img1.repeat([self.tile_z_dim, 1, 1]), img2.repeat([self.tile_z_dim, 1, 1])
                    imgori = imgori.repeat([self.tile_z_dim, 1, 1])

                sample['img1'] = img1
                sample['img2'] = img2
                sample['lb'] = lb
                sample['ori'] = imgori

           
         


            else:
                print("augtype error!!!")


        return sample

    def __len__(self):
        return len(self.actual_dataset)


def convert_to_png(img,low_num,high_num):
    x = np.array([low_num*1.0,high_num * 1.0])
    newimg = (img-x[0])/(x[1]-x[0])  
    newimg = (newimg*255).astype('uint8') 
    return newimg



def get_training(modality,norm_func, opt):
    print("get_train abd:",modality)
    if  opt.aug_type=='augseg_two_concat':
        tr_func = trans.transform_with_label(trans.pre_aug)
    
    print("tr_func:",tr_func)
    return AbdominalDataset(mode = 'train',transforms = tr_func,domains = modality,base_dir = BASEDIR,extern_norm_fn =norm_func,opt=opt)

def get_trval(modality, norm_func, opt):
    print("get_trval abd:",modality)
    return AbdominalDataset(mode = 'trval',transforms = None,domains = modality,base_dir = BASEDIR,extern_norm_fn = norm_func,opt=opt)

def get_trtest(modality, norm_func, opt):
    print("get_trtest abd:",modality)
    return AbdominalDataset(mode = 'trtest',transforms = None,domains = modality,base_dir = BASEDIR,extern_norm_fn = norm_func,opt=opt)

def get_test(modality, norm_func,opt):
     print("get_test abd:",modality)
     return AbdominalDataset(mode = 'test',transforms = None,domains = modality,base_dir = BASEDIR,extern_norm_fn = norm_func,opt=opt)
