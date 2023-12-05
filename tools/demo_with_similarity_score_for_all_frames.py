import glob
import warnings
import cv2
import numpy as np
import torch
from os.path import isfile, join
import argparse
from tools.test_dt import *
from custom import Custom

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

# Parse command line arguments
parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')
parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/tennis', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()

# Function to calculate similarity score between two frames
def calculate_similarity(frame1, frame2, siammask, target_pos, target_sz, device):
    state = siamese_init(frame1, target_pos, target_sz, siammask, cfg['hp'], device=device)  # Initialize tracker
    state = siamese_track(state, frame2, mask_enable=True, refine_enable=True, device=device)  # Track

    # Get the polygon and mask from the tracking state
    location = state['ploygon'].flatten()
    mask = state['mask'] > state['p'].seg_thr

    # Resize the mask to match the shape of frame2
    mask = cv2.resize(mask.astype(np.uint8), (frame2.shape[1], frame2.shape[0]))

    # Visualize the tracking result on the second frame
    frame2[:, :, 2] = (mask > 0) * 255 + (mask == 0) * frame2[:, :, 2]
    cv2.polylines(frame2, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
    cv2.imshow('SiamMask', frame2)
    cv2.waitKey(1)  # Use waitKey(1) to refresh the window

    # Return the similarity score
    return state['score']

if __name__ == '__main__':
    # Setup device (use GPU if available, otherwise use CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup the SiamMask model
    cfg = load_config(args)
    siammask = Custom(anchors=cfg['anchors'])

    # Load pre-trained weights if provided
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    # Set the model to evaluation mode and move it to the selected device (GPU or CPU)
    siammask.eval().to(device)

    # Parse image files from the specified directory
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))

    # Assume you have n frames
    n_frames = len(img_files)

    # Initialize variables for the initialization frame
    init_frame = cv2.imread(img_files[0])
    init_rect = cv2.selectROI('SiamMask', init_frame, False, False)
    x, y, w, h = init_rect
    target_pos = np.array([x + w / 2, y + h / 2])
    target_sz = np.array([w, h])

    # Calculate similarity scores for each frame compared to the initialization frame
    similarity_scores = []
    for i in range(1, n_frames):  # Start from the second frame
        current_frame = cv2.imread(img_files[i])
        similarity_score = calculate_similarity(init_frame, current_frame, siammask, target_pos, target_sz, device)
        similarity_scores.append(similarity_score)
        print(f"Similarity Score (Frame {i + 1}): {similarity_score}")

    # Print the overall similarity scores
    #print("All Similarity Scores:", similarity_scores)
    cv2.destroyAllWindows()  # Close the OpenCV windows when done
