import glob
import warnings
import cv2
import numpy as np
import torch
from os.path import isfile, join, basename
from natsort import natsorted  # Import natsorted for natural alphanumeric sorting
import argparse
from tools.testt import *
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
    return frame2, state['score']

# ... (previous code remains unchanged)

if __name__ == '__main__':
    # ... (previous code remains unchanged)
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
    # Initialize variables for the initialization frame
    init_frame_idx = 0
    init_rect = None
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    n_frames = len(img_files)
    # Create a single window
    cv2.namedWindow('SiamMask', cv2.WINDOW_NORMAL)

    # Calculate similarity scores for each frame compared to the initialization frame
    similarity_scores = []
    for i in range(n_frames):  # Start from the first frame
        current_frame = cv2.imread(img_files[i])

        if i == 0:
            init_frame = current_frame
            x = 153
            y = 126
            w=90
            h=51
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])

        if i - 10 >= 0:  # Check the condition inside the loop
            similarity_score_original = calculate_similarity(init_frame, current_frame, siammask, target_pos, target_sz, device)[1]
            similarity_scores.append(similarity_score_original)
            print(f"Similarity Score (Frame {i + 1}): {similarity_score_original}")

            # Check if similarity score falls below 0.7
            if similarity_score_original < 0.7:
                init_frame_idx = i - 10  # Set the initial frame to be two frames before the falling frame
                init_frame = cv2.imread(img_files[init_frame_idx])
                print(f"Changing Initial Frame to Frame {init_frame_idx + 1}")

        # Display the tracking result in the single window
        tracked_frame, _ = calculate_similarity(init_frame, current_frame, siammask, target_pos, target_sz, device)
        cv2.imshow('SiamMask', tracked_frame)
        cv2.waitKey(1)  # Use waitKey(1) to refresh the window

    # Print the overall similarity scores
    print("All Similarity Scores:", similarity_scores)
    cv2.destroyAllWindows()  # Close the OpenCV windows when done
