from __future__ import division
import argparse
import logging
import numpy as np
import cv2
from PIL import Image
from os import makedirs
from os.path import join, isdir, isfile
from tools.tracking_dt import *
from utils.log_helper import init_log, add_file_handler
from utils.load_helper import load_pretrain
from utils.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect
from utils.benchmark_helper import load_dataset, dataset_zoo
import subprocess
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from utils.anchors import Anchors
from utils.tracker_config import TrackerConfig

from utils.config_helper import load_config
from utils.pyvotkit.region import vot_overlap, vot_float2str

thrs = np.arange(0.3, 0.5, 0.05)

parser = argparse.ArgumentParser(description='Test SiamMask')
parser.add_argument('--arch', dest='arch', default='', choices=['Custom',],
                    help='architecture of pretrained model')
parser.add_argument('--config', dest='config', required=True, help='hyper-parameter for SiamMask')
parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--mask', action='store_true', help='whether use mask output')
parser.add_argument('--refine', action='store_true', help='whether use mask refine output')
parser.add_argument('--dataset', dest='dataset', default='VOT2018', choices=dataset_zoo,
                    help='datasets')
parser.add_argument('-l', '--log', default="log_test.txt", type=str, help='log file')
parser.add_argument('-v', '--visualization', dest='visualization', action='store_true',
                    help='whether visualize result')
parser.add_argument('--save_mask', action='store_true', help='whether use save mask for davis')
parser.add_argument('--gt', action='store_true', help='whether use gt rect for davis (Oracle)')
parser.add_argument('--video', default='', type=str, help='test special video')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
parser.add_argument('--debug', action='store_true', help='debug mode')

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  
    img = to_torch(img).float()
    return img

def calculate_direction(past_positions):
    positions = np.array(past_positions)
    differences = positions[1:] - positions[:-1]
    avg_direction = np.mean(differences, axis=0)
    avg_direction /= np.linalg.norm(avg_direction)
    return avg_direction

def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode='torch'):
    #preprocessing step
    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    #calculate the context boundaries to track based on position and size
    c = (original_sz + 1) / 2
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + sz - 1
    #padding
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))
    #adjusting context boundary based upon the padding
    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    r, c, k = im.shape
    #creating new image te_im with padding 
    #fills with average pixel values in the padded areas
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    #resizing
    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original
    
    return im_to_torch(im_patch) if out_mode in 'torch' else im_patch


def generate_anchor(cfg, score_size):
    anchors = Anchors(cfg) #initialize anchor object
    anchor = anchors.anchors # retrieving coordinates from anchors object
    #reshaping anchor coordinates for each corener
    x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
    #modified coordinates => x-center, y-center, width, height
    anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
    #retrieving total stride value from object
    #refers to the spacing between the anchor boxes placed across the image grid.
    total_stride = anchors.stride
    #determines number of anchor boxes
    anchor_num = anchor.shape[0]
    #score_size - used to calculate how many grids to draw to test
    #this function uses score_size to get bunch of guesses across the image
    """
    The center points of the replicated anchor boxes change relative to the 
    grid positions but maintain the same relative placement within their individual box.

    For example, if the initial anchor box's center is at coordinates (x, y),
    each replicated anchor box will be positioned at different coordinates (x', y') across the image grid,
    maintaining the same relative structure and proportions but distributed across the grid.
    """
    #reshaping the tile or array into 2d
    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    #below code assigns the positions for the anchors create in the above line
    #origin - generally the top-left corner of the grid
    ori = - (score_size // 2) * total_stride
    #grid of x and y coordinate based on the ori and total_stride
    #this grid spans across the image and covers different spatial locations
    #xx and yy r 2 matrices having x and y coordinates for every point in the grid
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    #flattening
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


def siamese_init(im, target_pos, target_sz, model, hp=None, device='cpu'):
    state = dict()
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]
    p = TrackerConfig()
    p.update(hp, model.anchors)

    p.renew()

    net = model
    p.scales = model.anchors['scales']
    p.ratios = model.anchors['ratios']
    p.anchor_num = model.anchor_num
    p.anchor = generate_anchor(model.anchors, p.score_size)
    #computes avg of red, blue, green of each pixel
    avg_chans = np.mean(im, axis=(0, 1))

    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z)) #original size for the subwindow

    #extracts a subwindow from the input image im based on target position
    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)
    #z is used as a template to siamese network
    z = Variable(z_crop.unsqueeze(0))
    #template formation for siamese network model
    net.template(z.to(device))
    #simply filteration
    #cosine - uses hanning window
    #window created by tapering smoothly from the center towards the edges
    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    #window created as uniform across the entire image patch
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)

    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['past_positions'] = [target_pos] * 9
    return state

def siamese_track(state, im, mask_enable=False, refine_enable=False, device='cpu', debug=False):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']
    #calculating width and height with context
    wc_x = target_sz[1] + p.context_amount * sum(target_sz)
    hc_x = target_sz[0] + p.context_amount * sum(target_sz)
    #calculates size with above ones i.e., indirectly with context
    s_x = np.sqrt(wc_x * hc_x)
    #scale computation
    scale_x = p.exemplar_size / s_x
    #search region & padding computation
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search/scale_x
    s_x = s_x + 2 * pad
    #defining search region
    crop_box = [target_pos[0] - round(s_x) / 2, target_pos[1] - round(s_x) / 2, round(s_x), round(s_x)]

    if debug:
        im_debug = im.copy()
        crop_box_int = np.int0(crop_box)
        cv2.rectangle(im_debug, (crop_box_int[0], crop_box_int[1]),
                      (crop_box_int[0] + crop_box_int[2], crop_box_int[1] + crop_box_int[3]), (255, 0, 0), 2)
        cv2.imshow('search area', im_debug)
        cv2.waitKey(0)
    
    #extracting a subwindow for tracking
    x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))
    #tracking of object for the given window from the input frame im
    #score, delta, mask could be arrays with different confidence scores for different postionfs
    #delta value marks the position changes
    if mask_enable:
        score, delta, mask = net.track_mask(x_crop.to(device))
    else:
        score, delta = net.track(x_crop.to(device))
    #adjusts the prediction of position and size changes i.e., delta
    #by applying scaling, shifting and exponential tranformations
    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    #softmax gives the vector for conditional probablity of each class
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0), dim=1).data[:,
            1].cpu().numpy()
    #adjusts the x axis postion change by scaling and shifting based on the anchor box values
    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    # for y axis
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    # for width using exponential scaling
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    # for  height using exponential scaling
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]
    #ensuring specific range
    def change(r):
        return np.maximum(r, 1. / r)
    #calcs the size of rectangle
    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)
    #given the tuple wh, calcs the size of rectangle
    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)
    #resized size reps target size within the cropped image patch
    target_sz_in_crop = target_sz*scale_x
    #computes the change factor in width and height compared to the size of the target within the cropped image patch
    #size change factor with respect to the neural network prediction
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz_in_crop))) 
    #based on ratio between width to height changes to the changes predicted in neural network
    r_c = change((target_sz_in_crop[0] / target_sz_in_crop[1]) / (delta[2, :] / delta[3, :]))  
    #calculatin penaly on score influence
    penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
    #calculates the penalty score with original score
    pscore = penalty * score
    #combining pscore and window(filtering) gives final weighted score
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    best_pscore_id = np.argmax(pscore)
    #predicted changes in pos and size from delta array
    pred_in_crop = delta[:, best_pscore_id] / scale_x
    #lr will be used in subsequent steps for tracking
    #controls the magnitude of updates applies to the estimate position and size of tracked object
    #reflects the algo's confidence in current prediction
    #adjust the impact of this prediction on future updates
    #cautious adjustments based on the quality of predictions
    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr
    #updated target position and size based on predicted change & current value
    res_x = pred_in_crop[0] + target_pos[0]
    res_y = pred_in_crop[1] + target_pos[1]

    res_w = target_sz[0] * (1 - lr) + pred_in_crop[2] * lr
    res_h = target_sz[1] * (1 - lr) + pred_in_crop[3] * lr
    # update target pos and size
    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])
    #for mask
    if mask_enable:
        best_pscore_id_mask = np.unravel_index(best_pscore_id, (5, p.score_size, p.score_size))
        delta_x, delta_y = best_pscore_id_mask[2], best_pscore_id_mask[1]
        if refine_enable:
            mask = net.track_refine((delta_y, delta_x)).to(device).sigmoid().squeeze().view(
                p.out_size, p.out_size).cpu().data.numpy()
        else:
            mask = mask[0, :, delta_y, delta_x].sigmoid(). \
                squeeze().view(p.out_size, p.out_size).cpu().data.numpy()
        def crop_back(image, bbox, out_sz, padding=-1):
            a = (out_sz[0] - 1) / bbox[2]
            b = (out_sz[1] - 1) / bbox[3]
            c = -a * bbox[0]
            d = -b * bbox[1]
            mapping = np.array([[a, 0, c],
                                [0, b, d]]).astype(np.float)
            crop = cv2.warpAffine(image, mapping, (out_sz[0], out_sz[1]),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=padding)
            return crop

        s = crop_box[2] / p.instance_size
        sub_box = [crop_box[0] + (delta_x - p.base_size / 2) * p.total_stride * s,
                   crop_box[1] + (delta_y - p.base_size / 2) * p.total_stride * s,
                   s * p.exemplar_size, s * p.exemplar_size]
        s = p.out_size / sub_box[2]
        back_box = [-sub_box[0] * s, -sub_box[1] * s, state['im_w'] * s, state['im_h'] * s]
        mask_in_img = crop_back(mask, back_box, (state['im_w'], state['im_h']))

        target_mask = (mask_in_img > p.seg_thr).astype(np.uint8)
        if cv2.__version__[-5] == '4':
            contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) != 0 and np.max(cnt_area) > 100:
            contour = contours[np.argmax(cnt_area)] 
            polygon = contour.reshape(-1, 2)
            prbox = cv2.boxPoints(cv2.minAreaRect(polygon))  
            rbox_in_img = prbox
        else:  
            location = cxy_wh_2_rect(target_pos, target_sz)
            rbox_in_img = np.array([[location[0], location[1]],
                                    [location[0] + location[2], location[1]],
                                    [location[0] + location[2], location[1] + location[3]],
                                    [location[0], location[1] + location[3]]])
    #checks the updated position and size stays within boundaries of image
    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    #retrieving and updateing past positions
    past_positions = state['past_positions']
    past_positions.pop(0)
    past_positions.append(target_pos.copy()) 
    #trend of movement
    direction_vector = calculate_direction(past_positions)
    direction_magnitude = np.linalg.norm(direction_vector)
    weight_factor = min(1.0, direction_magnitude * 0.1)
    #influence target position
    target_pos += direction_vector * weight_factor  

    initial_score = 0.1
    #updates score fields to best prediction score
    state['score']=score[best_pscore_id]
    similarity_score = state['score']
    if similarity_score > initial_score:
        lr = state['p'].lr * 1.1
    else:
        lr = state['p'].lr * 0.9

    state['past_positions'] = past_positions
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score[best_pscore_id]
    state['mask'] = mask_in_img if mask_enable else []
    state['ploygon'] = rbox_in_img if mask_enable else []
    return state
