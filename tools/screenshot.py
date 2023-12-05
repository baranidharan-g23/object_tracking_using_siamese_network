import glob
import warnings
import cv2
import numpy as np
import torch
from os.path import isfile, join, basename
from natsort import natsorted
import argparse
from tools.test import *
from custom import Custom

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')
parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/heli', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()

def calculate_similarity(frame1, frame2, siammask, target_pos, target_sz, device):
    state = siamese_init(frame1, target_pos, target_sz, siammask, cfg['hp'], device=device)
    state = siamese_track(state, frame2, mask_enable=True, refine_enable=True, device=device)
    location = state['ploygon'].flatten()
    mask = state['mask'] > state['p'].seg_thr
    mask = cv2.resize(mask.astype(np.uint8), (frame2.shape[1], frame2.shape[0]))
    frame2[:, :, 2] = (mask > 0) * 255 + (mask == 0) * frame2[:, :, 2]
    cv2.polylines(frame2, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
    return frame2, state['score']

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    cfg = load_config(args)
    siammask = Custom(anchors=cfg['anchors'])

    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)
    init_frame_idx = 0
    init_rect = None
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    n_frames = len(img_files)
    cv2.namedWindow('SiamMask', cv2.WINDOW_NORMAL)

    similarity_scores = []
    for i in range(n_frames):
        current_frame = cv2.imread(img_files[i])

        if i == 0:
            init_frame = current_frame
            init_rect = cv2.selectROI('SiamMask', init_frame, False, False)
            x, y, w, h = init_rect
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])

        if i - 10 >= 0:
            similarity_score_original = calculate_similarity(init_frame, current_frame, siammask, target_pos, target_sz, device)[1]
            similarity_scores.append(similarity_score_original)
            print(f"Similarity Score (Frame {i + 1}): {similarity_score_original}")

            if similarity_score_original < 0.7:
                init_frame_idx = i - 10
                init_frame = cv2.imread(img_files[init_frame_idx])
                print(f"Changing Initial Frame to Frame {init_frame_idx + 1}")

        tracked_frame, _ = calculate_similarity(init_frame, current_frame, siammask, target_pos, target_sz, device)
        cv2.imshow('SiamMask', tracked_frame)
        cv2.waitKey(1)

    print("All Similarity Scores:", similarity_scores)
    cv2.destroyAllWindows()
