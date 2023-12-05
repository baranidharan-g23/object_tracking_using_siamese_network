# Import necessary libraries and modules
import glob
from tools.test_dt import *

# Parse command line arguments
parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')
parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/tennis', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()

frame_counter = 0
template_update_interval = 10  # Update the template every 10 frames
initial_template = None
# Check if the script is being run as the main program
if __name__ == '__main__':
    # Setup device (use GPU if available, otherwise use CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup the SiamMask model
    cfg = load_config(args)
    from custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    
    # Load pre-trained weights if provided
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    # Set the model to evaluation mode and move it to the selected device (GPU or CPU)
    siammask.eval().to(device)

    # Parse image files from the specified directory
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    ims = [cv2.imread(imf) for imf in img_files]

    # Select the Region of Interest (ROI) using the first image
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    try:
        x = 153
        y = 126
        w=90
        h=51
        print("Initial ROI Coordinates (x, y, width, height):", x, y, w, h)
    except:
        exit()

    toc = 0
    # Loop through each frame in the sequence
    for f, im in enumerate(ims):
        tic = cv2.getTickCount()
        if f == 0:  # Initialization for the first frame
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # Initialize tracker
            initial_template = im[y:y+h, x:x+w]  
        elif f > 0:  # Tracking for subsequent frames
            state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # Track
            location = state['ploygon'].flatten()
            mask = state['mask'] > state['p'].seg_thr
            if state['score'] < 0.1:
            # Update the template to the initial one
                state = siamese_init(initial_template, target_pos, target_sz, siammask, cfg['hp'], device=device)

            # Visualize the tracking result on the image
            im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
            cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            cv2.imshow('SiamMask', im)
            key = cv2.waitKey(1)
            if key > 0:
                break
            frame_counter += 1
            if frame_counter == template_update_interval:
                frame_counter = 0
        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visualization!)'.format(toc, fps))
