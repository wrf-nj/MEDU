import os
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from DUALUNet import DUALUNet
import logging
import os.path as osp
import dataloaders.AbdominalDataset as ABD
import dataloaders.CardiacDataset as cardiac_cls
from PIL import Image
from segloss import Efficient_DiceScore,SoftDiceLoss,My_CE
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--hiera_path', type=str,  help='path to the sam2 pretrained hiera')
parser.add_argument('--save_path', type=str,  help='save path')
parser.add_argument('--expname', type=str,help='exp name')
parser.add_argument('--fseed', default=1, type=int)
parser.add_argument('--epoch', type=int, default=20, help='training epochs')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--batch_size', default=20, type=int)
parser.add_argument('--num_classes', type=int, default=2, help='num classes')
parser.add_argument('--dataset', type=str,  help='dataset name')
parser.add_argument('--tr_domain', type=str, help='tr_domain name')
parser.add_argument('--per', default=1.0, type=float)
parser.add_argument('--aug_type', default='augseg+geo', type=str)
parser.add_argument('--model_type', default='onlysam2unet', type=str)
parser.add_argument('--kl_w', default=1.0, type=float)
parser.add_argument('--con_f', default=0, type=int)
parser.add_argument('--dropoutflag', default=0, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)

args = parser.parse_args()

ScoreDiceEval = Efficient_DiceScore(args.num_classes, ignore_chan0=False).cuda()
criterionDice = SoftDiceLoss(args.num_classes).cuda()
criterionCE = My_CE(nclass=args.num_classes, batch_size=args.batch_size,weight=torch.ones(args.num_classes,)).cuda()
criterionCons=torch.nn.KLDivLoss()


def compute_kl_loss(p, q):
   

    eps = 1e-8
    loss_consistency = criterionCons(F.log_softmax(p, dim=1), F.softmax(q, dim=1)+eps) + criterionCons( F.log_softmax(q, dim=1), F.softmax(p, dim=1)+eps)

    loss_consistency =loss_consistency / 2.0
    return loss_consistency

def structure_loss(pred, mask,args):
   

    loss_dice= criterionDice(input = pred, target = mask)
    loss_ce= criterionCE(inputs = pred, targets =mask.long() )
    loss_all=loss_dice+loss_ce
   
    return loss_all




def pre_labmap():
    labmap={}
    tmp2={'0':0,'1':63,'2':126,'3':189,'4':255}
    labmap['ABDOMINAL']=tmp2
    tmp3={'0':0,'1':85,'2':170,'3':255}
    labmap['CARDIAC']=tmp3
    return labmap

def deal_wit_lbvis(tmp_mp,x,ncls):
   
    y=torch.zeros(size=x.shape).cuda()
    x=torch.from_numpy(x)
    for i in range(ncls):
        y[x==i]=tmp_mp[str(i)]
    return y


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def eval_model(test_loader,model,epoch,label_name,args,flag,savdir):
    with torch.no_grad():
        out_prediction_list = {}  # a buffer for saving results
        for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            if batch['is_start']:
                slice_idx = 0
                scan_id_full = str(batch['scan_id'][0])
                out_prediction_list[scan_id_full] = {}

                nframe = batch['nframe']
                nb, nc, nx, ny = batch['img1'].shape

                curr_pred = torch.Tensor(np.zeros([nframe, nx, ny])).cuda()  # nb/nz, nc, nx, ny
                curr_gth = torch.Tensor(np.zeros([nframe, nx, ny])).cuda()
                curr_img = torch.Tensor(np.zeros([nframe, nx, ny])).cuda()
            assert batch['lb'].shape[0] == 1  # enforce a batchsize of 1

            image,gth=batch['img1'].cuda(),batch['lb'].cuda()#  .to(device).to(device)

            model.eval()
            with torch.no_grad():
                pred, res1, res2 = model(image,dropoutflag=0)
                pred = torch.argmax(pred, 1)


          
            curr_pred[slice_idx, ...] = pred[0, ...]  # nb (1), nc, nx, ny
            curr_gth[slice_idx, ...] = gth[0, 0, ...]
            curr_img[slice_idx, ...] = batch['img1'][0, 1, ...]
            slice_idx += 1

            if batch['is_end']:
                out_prediction_list[scan_id_full]['pred'] = curr_pred
                out_prediction_list[scan_id_full]['gth'] = curr_gth
                out_prediction_list[scan_id_full]['img'] = curr_img


        print("Epoch {} test result on mode  seg:".format(epoch))

        score=eval_list_wrapper( out_prediction_list,  label_name,flag,savdir)
        torch.cuda.empty_cache()

    return out_prediction_list,score


def eval_list_wrapper(  vol_list,  label_name,flag,savdir):
    nclass = len(label_name)
    out_count = len(vol_list)  # is the former out_prediction_list
    tables_by_domain = {}  # tables by domain
    dsc_table = np.ones([out_count, nclass])  # rows and samples, columns are structures

    idx = 0
    for scan_id, comp in vol_list.items():
        domain, pid = scan_id.split("_")
        if domain not in tables_by_domain.keys():
            tables_by_domain[domain] = {'scores': [], 'scan_ids': []}
        pred_ = comp['pred']
        gth_ = comp['gth']

      

        dices = ScoreDiceEval(torch.unsqueeze(pred_,dim=1), gth_,dense_input=True)
        dices=dices.cpu().numpy()  # this includes the background class
        tables_by_domain[domain]['scores'].append([_sc for _sc in dices])
        tables_by_domain[domain]['scan_ids'].append(scan_id)
        dsc_table[idx, ...] = np.reshape(dices, (-1))
        del pred_
        del gth_
        idx += 1
        torch.cuda.empty_cache()

    error_dict = {}
    for organ in range(nclass):
        mean_dc = np.mean(dsc_table[:, organ])
        std_dc = np.std(dsc_table[:, organ])
        print("Organ {} with dice: mean: {} \n, std: {}".format(label_name[organ], mean_dc, std_dc))
        error_dict[label_name[organ]] = mean_dc
        if flag==1:
          with open(savdir+'/out.csv', 'a') as f:
            f.write("Organ"+label_name[organ] +"with dice: \n")
            f.write("mean:"+ str(mean_dc)+"\n")
            f.write("std:"+str(std_dc)+"\n")

        
        else:
          print("flag0")




    print("Overall mean dice by sample {}".format(
        dsc_table[:, 1:].mean()))  # background is noted as class 0 and therefore not counted
    error_dict['overall'] = dsc_table[:, 1:].mean()

    if flag==1:
        with open(savdir+'/out.csv', 'a') as f:
           f.write("Overall mean dice by sample:"+str(dsc_table[:,1:].mean())+" \n")

       
    else:
        print("flag0")


    # then deal with table_by_domain issue
    overall_by_domain = []
    domain_names = []
    for domain_name, domain_dict in tables_by_domain.items():
        domain_scores = np.array(tables_by_domain[domain_name]['scores'])
        domain_mean_score = np.mean(domain_scores[:, 1:])
        error_dict[f'domain_{domain_name}_overall'] = domain_mean_score
        error_dict[f'domain_{domain_name}_table'] = domain_scores
        overall_by_domain.append(domain_mean_score)
        domain_names.append(domain_name)

    error_dict['overall_by_domain'] = np.mean(overall_by_domain)
    print("!!!!!!!Overall mean dice by domain {}".format(error_dict['overall_by_domain']))
      # for prostate dataset, we use by-domain results to mitigate the differences in number of samples for each target domain
    if flag==1:
        with open(savdir + '/out.csv', 'a') as f:
            f.write("Overall mean dice by domain:" + str(error_dict['overall_by_domain']) + " \n")

       
    else:
        print("flag0")

    return error_dict['overall_by_domain']


def convert_to_png(img,low_num,high_num):
    x = np.array([low_num*1.,high_num * 1.])
    newimg = (img-x[0])/(x[1]-x[0]) 
    newimg = (newimg*255).astype('uint8')  
    return newimg




if __name__ == "__main__":
  

    script_path = os.path.abspath(sys.argv[0])
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "No Conda environment detected")
   

   


    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.fseed)
    np.random.seed(args.fseed)
    torch.manual_seed(args.fseed)
    torch.cuda.manual_seed(args.fseed)

    labmap = pre_labmap()
    labmap = labmap[args.dataset]
   
    if args.dataset== 'ABDOMINAL':
        if args.tr_domain == 'SABSCT':
            tr_domain = ['SABSCT']
            te_domain = ['CHAOST2']
        else:
            tr_domain = ['CHAOST2']
            te_domain = ['SABSCT']

        train_set = ABD.get_training(modality=tr_domain, norm_func=None, opt=args)
        tr_valset = ABD.get_trval(modality=tr_domain, norm_func=train_set.normalize_op,opt=args)  # not really using it as there is no validation for target
        tr_teset = ABD.get_trtest(modality=tr_domain, norm_func=train_set.normalize_op, opt=args)
        test_set = ABD.get_test(modality=te_domain, norm_func=None, opt=args)

        label_name = ABD.LABEL_NAME

    elif args.dataset== 'CARDIAC':
        if args.tr_domain == 'LGE':
            tr_domain = ['LGE']
            te_domain = ['bSSFP']
        else:
            tr_domain = ['bSSFP']
            te_domain = ['LGE']
        train_set = cardiac_cls.get_training(modality=tr_domain, opt=args)
        tr_valset = cardiac_cls.get_trval(modality=tr_domain, opt=args)
        tr_teset = cardiac_cls.get_trtest(modality=tr_domain, opt=args)  # as dataset split,cardiac didn't have this
        test_set = cardiac_cls.get_test(modality=te_domain, opt=args)

        label_name = cardiac_cls.LABEL_NAME

    else:
        print('not implement this dataset', args.dataset)

    train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True, worker_init_fn=worker_init_fn,pin_memory=True)
    trval_loader = DataLoader(dataset=tr_valset, num_workers=1, batch_size=1, shuffle=False, pin_memory=True)
    trte_loader = DataLoader(dataset=tr_teset, num_workers=1, batch_size=1, shuffle=False, pin_memory=True)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False, pin_memory=True)

   
    
    if args.model_type=='few_two_concat_linear':
       print("dualnet")
       print("args.model_type:",args.model_type)
       model =DUALUNet(args.hiera_path,ncls=args.num_classes,args=args)
       model=model.cuda()

    optim = opt.AdamW([{"params":model.parameters(), "initia_lr": args.lr}], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)

    save_path=args.save_path
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path+"/"+args.expname+"/", exist_ok=True)
    save_path=save_path+"/"+args.expname+"/"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    shutil.copytree('.', save_path + 'code', shutil.ignore_patterns(['.git', '__pycache__']))

    print("save_path:",save_path)
    os.makedirs(save_path + "imgs/", exist_ok=True)
    iters=0
    tbfile_dir = save_path+ "tboard/"
    logdir = save_path+"train/"
    if not os.path.exists(logdir):
        os.mkdir(logdir)
        os.mkdir(logdir+'img')
        os.mkdir(logdir+'pred')

    if not os.path.exists(tbfile_dir):
        os.mkdir(tbfile_dir)
    tb_writer = SummaryWriter(tbfile_dir)
    finalfile = logdir + 'out.csv'
    with open(finalfile, 'a') as f:
        f.write(args.expname+" \n")

   



    print("save_path logger:",save_path + "log.txt")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logger = logging.getLogger('log1')
    logging.basicConfig(filemode='a', level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s',datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.FileHandler(save_path+'log.txt', encoding='utf-8'))
    logging.info("config:"+str(args))
    logging.info("opt.f_seed:"+str(args.fseed))

    print("args.kl_w:",args.kl_w)
    for epoch in range(args.epoch):
        model.train()
        for i, batch in enumerate(train_loader):
            if args.aug_type == 'augseg_two_concat':
                print("augseg_two_concat")
                x1 = batch['img1']
                x1=x1.cuda()
                x2 = batch['img2']
                x2 = x2.cuda()
                x=torch.cat((x1, x2), dim=0)
                x = x.cuda()
                target = batch['lb']
                target = target.cuda()

       



            optim.zero_grad()


      
            print("dualnet")
            pred0, pred1, pred2 = model(x,args.dropoutflag)
            if args.aug_type == 'augseg_two_concat':
                print("loss augseg_two_concat")
                bs=target.shape[0]
                loss0 = structure_loss(pred0[:bs], target, args)
                loss1 = structure_loss(pred1[:bs], target, args)
                loss2 = structure_loss(pred2[:bs], target, args)
                loss = 2 * loss0 + 0.5 * loss1 + 0.5 * loss2

                if args.con_f==1:
                    print("args.con_f 1")
                    con0=compute_kl_loss(pred0[:bs],pred0[bs:])
                    con1 = compute_kl_loss(pred1[:bs], pred1[bs:])
                    con2 = compute_kl_loss(pred2[:bs], pred2[bs:])
                    con=(con0+con1+con2)/3.0
                    loss = loss + args.kl_w * con
                    
                



            loss.backward()
            optim.step()

            current_lr = scheduler.get_last_lr()
            current_lr=current_lr[0]
            print("epoch:{}-{}: lr:{} loss:{},{},{},{},{}".format(epoch + 1, i + 1,current_lr, loss.item(),loss0.item(),loss1.item(),loss2.item(),con.item()))
            logger.info("Tr-Epoch:{},Iter:{},Lr:{:.5f}--loss:{:5f} loss0:{:5f} loss1:{:5f} loss2:{:5f} con:{:5f}".format(epoch + 1, i + 1,current_lr, loss.item(),loss0.item(),loss1.item(),loss2.item(),con.item()))
            iters=iters+1

            tb_writer.add_scalar('lr', current_lr, iters)
            tb_writer.add_scalar('loss/loss', loss, iters)
           

        scheduler.step()

        if (epoch+1) % 750 == 0 or (epoch+1) == args.epoch:
            torch.save(model.state_dict(), os.path.join(save_path, 'ckpt-%d.pth' % (epoch + 1)))
            print('[Saving Snapshot:]', os.path.join(save_path, 'ckpt-%d.pth'% (epoch + 1)))

       


       
    print("train over,testing!!!!!!!!!")
  
    model2 =DUALUNet(args.hiera_path,ncls=args.num_classes,args=args)

    model2=model2.cuda()#.to(device)
    reload_model_fid = os.path.join(save_path, 'ckpt-%d.pth' % args.epoch )
    print("loading!:",reload_model_fid)

    model2.load_state_dict(torch.load(reload_model_fid), strict=True)

    with torch.no_grad():
        model2.eval()
        with open(finalfile, 'a') as f:
            f.write("test\n")
        _, te_score = eval_model(test_loader, model2, args.epoch, label_name, args,1,logdir)

       

