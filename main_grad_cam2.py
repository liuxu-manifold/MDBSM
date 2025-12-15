import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import datetime
import shutil
import sys
from pathlib import Path
# from utils.config import get_config
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import AverageMeter, reduce_tensor, epoch_saving, load_checkpoint, generate_text, auto_resume_helper
from datasets.build import build_dataloader
from utils.logger import create_logger
import time
import numpy as np
import random
from apex import amp
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets.blending import CutmixMixupBlending
# sys.path.append("/data/lql/zjl_86_torch/VideoX-master/X-CLIP/")
from utils.config import get_config
from sklearn.metrics import confusion_matrix
from PIL import Image
# from models import xclip as x_load
###pycharm训练
sys.path.append("/data/zjl/192-torch2/VideoX-master/X-CLIP/models/")
from xclip import load as x_load
import argparse
import cv2
import numpy as np
import torch
import  timm
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit

import datetime
from  tqdm import tqdm

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['RANK'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['WORLD_SIZE'] = '1'

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
# os.environ['MASTER_PORT'] = '5678'
# if not dist.is_initialized():
#     # dist.init_process_group(backend='nccl')
#
#     dist.init_process_group(backend='nccl', init_method='env://', rank = 0, world_size = 1)
# dist.init_process_group(backend='nccl')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=False, type=str,
                        default="/data/zjl/192-torch2/VideoX-master/X-CLIP/configs/k400/32_8.yaml")
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--output', type=str, default="/data/zjl/192-torch2/VideoX-master/X-CLIP/save_cam_img/")
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)

    parser.add_argument("--train_txt_path", type=str, default=None, help='txt file path of training data')
    parser.add_argument("--test_txt_path", type=str, default=None, help='txt file path of testing data')
    parser.add_argument("--save_txt_path", type=str, default=None, help='txt file path of testing data')

    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
    parser.add_argument("--fold_id", type=int, default=0, help='local rank for DistributedDataParallel')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    args = parser.parse_args()

    os.environ['MASTER_PORT'] = '12279'
    flg = 0
    print("use own frame:", bool(flg))
    args.local_rank = 0
    args.batch_size = 32
    flod_id = 1
    args.flod_id = flod_id
    ####MAFW_I
    args.train_txt_path = f"/data/zjl/192-torch/pytorch-coviar-master/data/datalists/MAFW-txt/set_{flod_id}_train.txt"
    args.test_txt_path = f"/data/zjl/192-torch/pytorch-coviar-master/data/datalists/MAFW-txt/set_{flod_id}_test.txt"
    ####DFER_I
    # args.train_txt_path = f"/data/zjl/192-torch/pytorch-coviar-master/data/datalists/DFEW-txt/DFEW_set_{flod_id}_train.txt"
    # args.test_txt_path = f"/data/zjl/192-torch/pytorch-coviar-master/data/datalists/DFEW-txt/DFEW_set_{flod_id}_test.txt"


    # args.train_txt_path = f"/home/zjl/pytorch-coviar-master/data/datalists/DFEW-txt/DFEW_set_{flod_id}_train.txt"
    # args.test_txt_path = f"/home/zjl/pytorch-coviar-master/data/datalists/DFEW-txt/DFEW_set_{flod_id}_test.txt"
    # args.test_txt_path = f"/data/lql/zjl_86_torch/Erasing-Attention-Consistency-main/datalist/DFEW-txt/DFEW_set_{flod_id}_test.txt"
    # args.train_txt_path = f"/data/lql/zjl_86_torch/Erasing-Attention-Consistency-main/datalist/MAFW-txt/set_{flod_id}_train.txt"
    # args.test_txt_path = f"/data/lql/zjl_86_torch/Erasing-Attention-Consistency-main/datalist/MAFW-txt/set_{flod_id}_test.txt"

    # args.train_txt_path = "/data/lql/zjl_86_torch/Erasing-Attention-Consistency-main/datalist/FER39k-txt/train.txt"
    # args.test_txt_path = "/data/lql/zjl_86_torch/Erasing-Attention-Consistency-main/datalist/FER39k-txt/test.txt"

    # args.train_txt_path = f"/data/lql/zjl_86_torch/Erasing-Attention-Consistency-main/datalist/Oulu-random/oulu-10-{flod_id}/oulu-train-10-{flod_id}.txt"
    # args.test_txt_path = f"/data/lql/zjl_86_torch/Erasing-Attention-Consistency-main/datalist/Oulu-random/oulu-10-{flod_id}/oulu-test-10-{flod_id}.txt"

    args.save_txt_path = f"/data/zjl/192-torch2/VideoX-master/X-CLIP/save_cam_img/"

    data_name = args.test_txt_path.split('/')[-2].split('-')[0]
    data_name_path = os.path.join(args.save_txt_path, data_name)
    if not os.path.exists(data_name_path):
        os.makedirs(data_name_path)
    now = datetime.datetime.now()
    current_date = now.strftime("%m-%d")
    current_time = now.strftime("%H-%M-%S")

    # 构建带有日期和时间的文件名
    file_name = f"{data_name}_第{flod_id}折_{current_date}日_{current_time}时.txt"
    dir_name = file_name.split('.')[0]
    data_name_path = os.path.join(data_name_path, dir_name)
    if not os.path.exists(data_name_path):
        os.makedirs(data_name_path)
    args.save_txt_path = os.path.join(data_name_path, file_name)

    config = get_config(args)

    return args, config


def VisualizationForOneDomain(Feature_Target, Label_Target, config, epoch):
    '''Feature Visualization in Source and Target Domain.'''

    Feature_Target = np.vstack(Feature_Target)
    Label_Target = np.concatenate(Label_Target)

    Feature = Feature_Target
    Label = Label_Target

    # Using T-SNE
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=50, early_exaggeration=3, )
    embedding = tsne.fit_transform(Feature)

    # Draw Visualization of Feature
    colors = {0: 'red', 1: 'blue', 2: 'olive', 3: 'green', 4: 'orange', 5: 'purple', 6: 'darkslategray'}
    labels = {0: 'Surprised', 1: 'Fear', 2: 'Disgust', 3: 'Happy', 4: 'Sad', 5: 'Angry', 6: 'Neutral'}

    data_min, data_max = np.min(embedding, 0), np.max(embedding, 0)
    data_norm = (embedding - data_min) / (data_max - data_min)

    fig = plt.figure()
    # ax = plt.subplot(111)

    for i in range(7):

        data_target_x, data_target_y = data_norm[Label == i][:, 0], data_norm[Label == i][:, 1]
        target_scatter = plt.scatter(data_target_x, data_target_y, color=colors[i], edgecolor="none", s=30,
                                     label=labels[i], marker="o", alpha=0.6, linewidth=0.1)

        if i == 0:
            target_legend = target_scatter

    save_tsne_path = os.path.join(os.path.dirname(config.DATA.TXT_FILE), f'epoch_{epoch}.png')
    plt.savefig(fname=save_tsne_path, format="png", bbox_inches='tight')

def reshape_transform(tensor, height=14, width=14):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 2:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result
def get_idx_img(video_path, idx):


    image_list = os.listdir(video_path)
    image_list.sort(key=lambda x: int(x.replace("FirstF_", "").split('.')[0]))

    # image_list = sorted(image_list)
    image_name = image_list[idx]
    image_name_path = os.path.join(video_path, image_name)

    image = Image.open(image_name_path)
    image = image.resize((224, 224))
    image = np.array(image)
    image = image/ np.max(image)
    image = image[..., ::-1]
    return image

def save_idx_ca_img(video_path, idx, cam_image, save_path, label_id_id):
    last_folder_name = os.path.basename(os.path.normpath(video_path))

    # save_path = os.path.join(save_path, last_folder_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = f'{label_id_id}_{last_folder_name}_{idx}_cam.jpg'
    # save_name = f'{last_folder_name}_{idx}_cam.jpg'
    save_path = os.path.join(save_path, save_name)
    cv2.imwrite(save_path, cam_image)


def main(config, args):
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}


    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    model, _ = x_load(config.MODEL.PRETRAINED, config.MODEL.ARCH,
                      device="cpu", jit=False,
                      T=config.DATA.NUM_FRAMES,
                      droppath=config.MODEL.DROP_PATH_RATE,
                      use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                      use_cache=config.MODEL.FIX_TEXT,
                      logger=logger,
                      )
    model = model.cuda()

    target_layers = [model.visual.transformer.resblocks[-10].ln_1]
    cam = methods['gradcam'](model=model,
                               target_layers=target_layers,
                               # use_cuda=True,
                               reshape_transform=reshape_transform)
    cam.batch_size = config.TRAIN.BATCH_SIZE

    text_labels = generate_text(train_data)
    text_inputs = text_labels.cuda()
    # mode1_dict = model.state_dict()
    # pre_dict = torch.load(
    #     "/data/lql/zjl_86_torch/VideoX-master/X-CLIP/output/chekpoint/DFEW_第1折_04-04日_02-45-43时/best.pth",map_location='cuda:6')[
    #                           'model']


    #text
    # model.load_state_dict(torch.load(
    #        "/data/lql/zjl_86_torch/VideoX-master/X-CLIP/output/chekpoint/DFEW_第1折_04-04日_02-40-36时/best.pth",map_location=f'cuda:{config.DATA.FlOD}')[
    #                           'model'])



    model.eval()
    # save_path = "/data/lql/zjl_86_torch/VideoX-master/X-CLIP/output/save_cam_img/fc/fc_T_6/"
    save_path = "/data/zjl/192-torch2/VideoX-master/X-CLIP/save_cam_img/"
    for idx, batch_data in tqdm(enumerate(val_loader)):
        _image = batch_data["imgs"]
        label_id = batch_data["label"]
        label_id_id = label_id.tolist()[0]
        video_path = batch_data["filename"][0]
        frame_inds = batch_data["frame_inds"].tolist()[0]
        # frame_inds = [1, 2, 3, 4, 5, 6, 7, 8]
        # frame_inds = [9,10,11,12,13,14,15,16]
        # frame_inds = [17,18,19,20,21,22,23,24]
        # frame_inds = [25,26,27,28,29,30,31,32]
        # frame_inds = [33,34,35,36,37,38,39,40]
        # frame_inds = [41,42,43,44,45,46,47,48]
        # frame_inds = [49,50,51,52,53,54,55,56]
        # frame_inds = [57,58,59,60,61,62,63,64]
        # frame_inds = [65,66,67,68,69,70,71,72]
        # frame_inds = [73,74,75,76,77,78,79,80]
        # frame_inds = [81,82,83,84,85,86,87,88]
        # frame_inds = [89,90,91,92,93,94,95,96]
        # frame_inds = [1,29,38,53,73,75,77,84]
        # frame_inds = [57,58,59,60,61,62,62,62]
        label_id = label_id.reshape(-1)
        b, tn, c, h, w = _image.size()
        t = config.DATA.NUM_FRAMES
        n = tn // t
        _image = _image.view(b, n, t, c, h, w)

        tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
        for i in range(n):
            image = _image[:, i, :, :, :, :]  # [b,t,c,h,w]
            label_id = label_id.cuda(non_blocking=True)
            image_input = image.cuda(non_blocking=True)

            grayscale_cam = cam(input_tensor=image_input,
                                text_tensor = text_inputs,
                                targets=None,
                                eigen_smooth=args.eigen_smooth,
                                aug_smooth=args.aug_smooth)

            grayscale_cam = grayscale_cam.reshape(b, t, h, w)
        for i in range(b):
            video_sequence = image[i,:,:,:,:]
            video_grayscale_cam = grayscale_cam[i,:,:,:]
            for j in range(t):

                video_image_cam = video_grayscale_cam[j,:,:]
                idx = frame_inds[j]

                video_image = get_idx_img(video_path, idx)

                cam_image = show_cam_on_image(video_image, video_image_cam)
                save_idx_ca_img(video_path, idx, cam_image, save_path, label_id_id)


    print("Finish!!!")







            # output = model(image_input, text_inputs)












    mixup_fn = None
    if config.AUG.MIXUP > 0:
        criterion = SoftTargetCrossEntropy()
        mixup_fn = CutmixMixupBlending(num_classes=config.DATA.NUM_CLASSES,
                                       smoothing=config.AUG.LABEL_SMOOTH,
                                       mixup_alpha=config.AUG.MIXUP,
                                       cutmix_alpha=config.AUG.CUTMIX,
                                       switch_prob=config.AUG.MIXUP_SWITCH_PROB)
    elif config.AUG.LABEL_SMOOTH > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
    else:
        criterion = nn.CrossEntropyLoss()
    # name_list = []
    # param_list = []
    # for name, param in model.named_parameters():
    #     if 'fta_' in name:
    #         name_list.append(name)
    #         param_list.append(param)
    #     print(name_list)

    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    if config.TRAIN.OPT_LEVEL != 'O0':
        model, optimizer = amp.initialize(models=model, optimizers=optimizer, opt_level=config.TRAIN.OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False,
                                                      find_unused_parameters=True)

    start_epoch, max_accuracy = 0, 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')

            with open(config.DATA.TXT_FILE, 'a+') as f:
                f.write(f'auto resuming from {resume_file}\n')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

            with open(config.DATA.TXT_FILE, 'a+') as f:
                f.write(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume\n')

    if config.MODEL.RESUME:
        start_epoch, max_accuracy = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)

    text_labels = generate_text(train_data)

    if config.TEST.ONLY_TEST:
        acc1 = validate(val_loader, text_labels, model, config)
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")
        with open(config.DATA.TXT_FILE, 'a+') as f:
            f.write(f'Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%\n')
        return
    current_UAR = 0.0
    best_UAR = 0.0
    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn)
        acc1, UAR, cm, Feature_Target, Label_Target = validate(val_loader, text_labels, model, config)
        VisualizationForOneDomain(Feature_Target, Label_Target, config, epoch)

        # acc1, UAR, cm = validate(val_loader, text_labels, model, config)
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")
        with open(config.DATA.TXT_FILE, 'a+') as f:
            f.write(f'Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%\n')
        is_best = acc1 > max_accuracy
        # max_accuracy = max(max_accuracy, acc1)
        if acc1 > max_accuracy:
            max_accuracy = acc1
            current_UAR = UAR
            print_confusion_matrix(cm, config, 'current_UAR')
        if acc1 == max_accuracy and UAR > current_UAR:
            current_UAR = max(current_UAR, UAR)
            print_confusion_matrix(cm, config, 'current_UAR')

        if UAR > best_UAR:
            best_UAR = UAR
            # print_confusion_matrix(cm, config, 'best_UAR')

        logger.info(
            f'Max accuracy: {max_accuracy:.2f}%, Current UAR : {current_UAR * 100:.2f}%, Max UAR :{best_UAR * 100:.2f}%')
        with open(config.DATA.TXT_FILE, 'a+') as f:
            f.write(
                f'Max accuracy: {max_accuracy:.2f}%, Current UAR : {current_UAR * 100:.2f}%, Max UAR :{best_UAR * 100:.2f}%')
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            epoch_saving(config, epoch, model.module, max_accuracy, optimizer, lr_scheduler, logger,
                         config.DATA.TXT_FILE,
                         is_best)

    config.defrost()
    config.TEST.NUM_CLIP = 4
    config.TEST.NUM_CROP = 3
    config.freeze()
    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    acc1 = validate(val_loader, text_labels, model, config)
    logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")
    with open(config.DATA.TXT_FILE, 'a+') as f:
        f.write(f'Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%\n')


def print_confusion_matrix(conf_matrix, config, name):
    save_cm_root = os.path.dirname(config.DATA.TXT_FILE)
    save_name = f'{config.DATA.FLOD}-{name}.png'
    save_path = os.path.join(save_cm_root, save_name)
    conf_matrix = torch.from_numpy(conf_matrix)
    """draw confusion matrix

    Args:
        data (dict): Contain config data
        path (path-like): The path to save picture
    """
    conf_matrix = conf_matrix / torch.sum(conf_matrix, dim=1, keepdim=True)
    # draw
    plt.figure()

    plt.imshow(conf_matrix, cmap=plt.cm.Blues)  # 可以改变颜色
    plt.colorbar()
    # labels = ["Ha", 'Sa', 'Ne', 'Ar', 'Su', 'Disg', 'Fe', 'Co', 'An', 'He', 'Disa']
    labels = ["Ha", 'Sa', 'Ne', 'Ar', 'Su', 'Di', 'Fe']  # 每种类别的标签
    # labels = ["Ha", 'Sa', 'Ne', 'Ar', 'Su', 'Di']  # 每种类别的标签
    indices = list(range(config.DATA.NUM_CLASSES))  ###更改类别
    plt.xticks(indices, labels, rotation=45)
    plt.yticks(indices, labels)
    # plt.xlabel('pred')
    # plt.ylabel('true')
    # 显示数据
    for first_index in range(config.DATA.NUM_CLASSES):  # trues###更改类别
        for second_index in range(config.DATA.NUM_CLASSES):  # preds###更改类别
            if conf_matrix[second_index][first_index] < 0.55:
                plt.text(first_index, second_index,
                         "{:.2f}".format(conf_matrix[second_index][first_index].item() * 100),
                         verticalalignment='center', horizontalalignment='center', fontsize=13)
            else:
                plt.text(first_index, second_index,
                         "{:.2f}".format(conf_matrix[second_index][first_index].item() * 100),
                         verticalalignment='center', horizontalalignment='center', color='white', fontsize=13)

    plt.tight_layout()
    plt.savefig(fname=save_path, format="png")
    plt.close()


def get_UAR(trues_te, pres_te):
    cm = confusion_matrix(trues_te, pres_te)
    # print("cm.size=",cm.size())
    acc_per_cls = [cm[i, i] / sum(cm[i]) for i in range(len(cm))]
    UAR = sum(acc_per_cls) / len(acc_per_cls)
    return UAR, cm


def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn):
    model.train()
    optimizer.zero_grad()

    num_steps = len(train_loader)
    batch_time = AverageMeter()
    tot_loss_meter = AverageMeter()

    start = time.time()
    end = time.time()

    texts = text_labels.cuda(non_blocking=True)
    trues_te = []
    for idx, batch_data in enumerate(train_loader):

        images = batch_data["imgs"].cuda(non_blocking=True)
        label_id = batch_data["label"].cuda(non_blocking=True)
        label_id = label_id.reshape(-1)
        images = images.view((-1, config.DATA.NUM_FRAMES, 3) + images.size()[-2:])

        if mixup_fn is not None:
            images, label_id = mixup_fn(images, label_id)

        if texts.shape[0] == 1:
            texts = texts.view(1, -1)

        criterion2 = nn.CrossEntropyLoss()
        # output , output2 = model(images, texts)
        output = model(images, texts)
        total_loss = criterion(output, label_id)
        # total_loss = criterion(output, label_id) + criterion(output2, label_id)*2
        total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS

        if config.TRAIN.ACCUMULATION_STEPS == 1:
            optimizer.zero_grad()
        if config.TRAIN.OPT_LEVEL != 'O0':
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        tot_loss_meter.update(total_loss.item(), len(label_id))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

            with open(config.DATA.TXT_FILE, 'a+') as f:
                f.write(f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                        f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
                        f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                        f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
                        f'mem {memory_used:.0f}MB\n')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    with open(config.DATA.TXT_FILE, 'a+') as f:
        f.write(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}\n")


@torch.no_grad()
def validate(val_loader, text_labels, model, config):
    model.eval()
    Feature_Target, Label_Target = [], []
    acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
    pres_te, trues_te = [], []
    with torch.no_grad():
        text_inputs = text_labels.cuda()
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        with open(config.DATA.TXT_FILE, 'a+') as f:
            f.write(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference\n")
        for idx, batch_data in enumerate(val_loader):
            _image = batch_data["imgs"]
            label_id = batch_data["label"]
            label_id = label_id.reshape(-1)

            b, tn, c, h, w = _image.size()
            t = config.DATA.NUM_FRAMES
            n = tn // t
            _image = _image.view(b, n, t, c, h, w)

            tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
            for i in range(n):
                image = _image[:, i, :, :, :, :]  # [b,t,c,h,w]
                label_id = label_id.cuda(non_blocking=True)
                image_input = image.cuda(non_blocking=True)

                if config.TRAIN.OPT_LEVEL == 'O2':
                    image_input = image_input.half()

                # output , output2 = model(image_input, text_inputs)
                output = model(image_input, text_inputs)
                similarity = output.view(b, -1).softmax(dim=-1)
                tot_similarity += similarity

            values_1, indices_1 = tot_similarity.topk(1, dim=-1)
            values_5, indices_5 = tot_similarity.topk(5, dim=-1)
            Feature_Target.append(output.cpu().data.numpy())
            Label_Target.append(label_id.cpu().data.numpy())

            trues_te += label_id.cpu().numpy().tolist()
            acc1, acc5 = 0, 0
            for i in range(b):
                pres_te += indices_1[i].cpu().numpy().tolist()
                if indices_1[i] == label_id[i]:
                    acc1 += 1
                if label_id[i] in indices_5[i]:
                    acc5 += 1

            acc1_meter.update(float(acc1) / b * 100, b)
            acc5_meter.update(float(acc5) / b * 100, b)
            if idx % config.PRINT_FREQ == 0:
                logger.info(
                    f'Test: [{idx}/{len(val_loader)}]\t'
                    f'Acc@1: {acc1_meter.avg:.3f}\t'
                )
                with open(config.DATA.TXT_FILE, 'a+') as f:
                    f.write(f'Test: [{idx}/{len(val_loader)}]\t'
                            f'Acc@1: {acc1_meter.avg:.3f}\t\n')

    UAR_te, cm = get_UAR(trues_te, pres_te)
    acc1_meter.sync()
    acc5_meter.sync()
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f} UAR {UAR_te * 100:.3f}')
    with open(config.DATA.TXT_FILE, 'a+') as f:
        f.write(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f} UAR {UAR_te * 100:.3f}')
    return acc1_meter.avg, UAR_te, cm, Feature_Target, Label_Target


if __name__ == '__main__':
    # prepare config
    args, config = parse_option()

    # new_cfg = config.clone()
    # config = new_cfg.clone()

    # flod_id = 1
    # args.local_rank = 7
    # config.DATA.TRAIN_FILE = f"/data/lql/zjl_86_torch/VideoX-master/X-CLIP/data/Oulu-random/oulu-10-{str(flod_id)}/oulu-train-10-{str(flod_id)}.txt"
    # config.DATA.VAL_FILE = f"/data/lql/zjl_86_torch/VideoX-master/X-CLIP/data/Oulu-random/oulu-10-{str(flod_id)}/oulu-test-10-{str(flod_id)}.txt"

    # init_distributed
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier(device_ids=[args.local_rank])

    # dist.init_process_group(backend='nccl', init_method='env://')
    # world_size = dist.get_world_size()
    # rank = dist.get_rank()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # create working_dir
    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)

    # logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    logger.info(f"working dir: {config.OUTPUT}")
    with open(config.DATA.TXT_FILE, 'a+') as f:
        f.write(f"working dir: {config.OUTPUT}\n")

    # save config
    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, config.OUTPUT)

    main(config, args)