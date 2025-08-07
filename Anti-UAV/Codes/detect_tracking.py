from __future__ import division

# Standard libraries
import os
import sys
import time
import struct
import socket
import logging
import datetime
import argparse
import pdb
import importlib

# Scientific and tensor libraries
import numpy as np
import torch
import cv2
from PIL import Image

# Add parent directory to path so "Codes" is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(__file__)
PKG1 = "Codes.detect_wrapper"
PKG2 = "Codes.tracking_wrapper"

sys.modules.setdefault("detect_wrapper",                   importlib.import_module(PKG1))
sys.modules.setdefault("detect_wrapper.models",            importlib.import_module(PKG1 + ".models"))
sys.modules.setdefault("detect_wrapper.utils",             importlib.import_module(PKG1 + ".utils"))

sys.modules.setdefault("tracking_wrapper",                 importlib.import_module(PKG2))
sys.modules.setdefault("tracking_wrapper.drtracker",       importlib.import_module(PKG2 + ".drtracker"))
sys.modules.setdefault("tracking_wrapper.dronetracker",    importlib.import_module(PKG2 + ".dronetracker"))
# ---- end bootstrap ----

from Codes.detect_wrapper.Detectoruav import DroneDetection
from Codes.tracking_wrapper.dronetracker.trackinguav.evaluation.tracker import Tracker

# All import paths have been consolidated at the top of the file
# We only need one path for the Anti-UAV directory, which contains the Codes module


import warnings
warnings.filterwarnings("ignore")



# # import torchvision
# # from torch.utils.data import DataLoader
# # from torchvision import datasets
# from torch.autograd import Variable
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
#from matplotlib.ticker import NullLocator
#import json


# Global initialization flags and objects
g_init = False         # Flag to track if system is initialized
g_detector = None      # Detector object
g_tracker = None       # Tracker object
g_logger = None        # Logger object
detect_box = None      # Current detection bounding box
track_box = None       # Current tracking bounding box
g_data = None          # Latest received UDP data
detect_first = True    # Flag to track first detection
g_enable_log = True    # Enable logging
repeat_detect = True   # Flag to repeat detection

# Frame counters
count = 0              # General frame counter
g_frame_counter = 0    # Counter for tracking frames
TRACK_MAX_COUNT = 150  # Maximum frames to track before re-detecting

# Feature flags
Visualization = 1      # Enable visualization (0=off, 1=on)
sendLocation = 0       # Enable sending location data (0=off, 1=on)

# Global UDP socket for sending coordinates
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Target host IP and port for sending coordinates
IP = '192.168.0.171'   # Target host IP
Port = '9921'          # Target host port

def safe_log(msg):
    """
    Safely log a message if logger is initialized.
    """
    if g_logger:
        g_logger.info(msg)


def send_bbs(bbs):
    """
    Log bounding box information if logger is initialized.
    """
    global g_logger
    if g_logger:
        g_logger.info('send a box : {}'.format(bbs))

def mono_to_rgb(data):
    """
    Convert a grayscale (mono) image to RGB by copying the single channel to all three channels.
    """
    w, h = data.shape
    img = np.zeros((w, h, 3), dtype=np.uint8)  # Create empty RGB image
    img[:, :, 0] = data                         # Copy to R channel
    img[:, :, 1] = data                         # Copy to G channel
    img[:, :, 2] = data                         # Copy to B channel
    return img

def rgb_to_ir(data):
    """
    Convert an RGB image to IR (infrared) format by extracting only the R channel.
    """
    w, h, c = data.shape
    img = data[:,:,0]  # Extract R channel only
    return img

def distance_check(bbx1, bbx2, thd):
    """
    Check if the Euclidean distance between centers of two bounding boxes is less than a threshold.
    """
    cx1 = bbx1[0]+bbx1[2]/2  # Center X of first bounding box
    cy1 = bbx1[1]+bbx1[3]/2  # Center Y of first bounding box
    cx2 = bbx2[0]+bbx2[2]/2  # Center X of second bounding box
    cy2 = bbx2[1]+bbx2[3]/2  # Center Y of second bounding box
    dist = np.sqrt((cx1-cx2)**2+(cy1-cy2)**2)  # Euclidean distance
    return dist<thd  # Return True if distance is less than threshold

def scale_coords(img1_shape, coords, img0_shape):
    """
    Rescale coordinates (xyxy) from img1_shape to img0_shape.
    """
    # Calculate scaling factors
    gainx = img1_shape[0] / img0_shape[0]
    gainy = img1_shape[1] / img0_shape[1]

    # Apply scaling to coordinates
    coords[0]= coords[0]/gainx  # x
    coords[1]= coords[1]/gainy  # y
    coords[2]= coords[2]/gainx  # width
    coords[3]= coords[3]/gainy  # height
    
    # Convert to integers
    coords = [int(x) for x in coords]
    return coords

def send_coord(coord):
    """
    Send target coordinates via UDP to the specified IP and port.
    """
    address = (IP, int(Port))
    # Define C struct: Target tracking information
    msgCode = 1
    nAimType = 1
    nTrackType = 1
    nState = 1
    
    if coord != None:
        # Extract coordinates
        nAimX = coord[0]  # x position
        nAimY = coord[1]  # y position
        nAimW = coord[2]  # width
        nAimH = coord[3]  # height
        
        # Pack data into binary struct
        data = struct.pack("iiiiiiii", msgCode, nAimType, nAimX, nAimY, nAimW, nAimH,nTrackType,nState)
        
        # Send via UDP
        udp_socket.sendto(data, address)
        safe_log("send successfully")
    else:
        # No coordinates to send
        pass
     
def global_init():
    """
    Initialize global variables for detection and tracking.
    
    This function:
    1. Sets up logging
    2. Checks CUDA availability
    3. Initializes the detector and tracker with appropriate model weights
    """
    global g_init, g_detector, g_tracker, g_logger, g_enable_log
    
    # Only initialize once
    if not g_init:
        # Check for CUDA availability and print information
        if torch.cuda.is_available():
            print("CUDA is available! Using GPU:", torch.cuda.get_device_name(0))
            print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
            device = torch.device("cuda")
        else:
            print("CUDA is not available. Using CPU.")
            device = torch.device("cpu")
        
        # Setup logging if enabled
        if g_enable_log:
            g_logger = logging.getLogger()
            g_logger.setLevel(logging.INFO)
            
            # Create directory for log file if it doesn't exist
            if not os.path.exists('c:/data'):
                os.makedirs('c:/data', exist_ok=True)
            
            # Create file handler for logging
            fh = logging.FileHandler('c:/data/log.txt', mode='a')
            g_logger.addHandler(fh)

        # Define model weights paths
        base = r"C:\Users\kinac\DroneDetection\runs\detect\subset36_run22\weights"
        IRweights_path  = os.path.join(base, "best.pt")
        RGBweights_path = IRweights_path  # TODO: Use different file if needed
        
        # Initialize detector and tracker
        g_detector = DroneDetection(IRweights_path=IRweights_path,
                                   RGBweights_path=RGBweights_path)
        g_tracker = Tracker()
        
        # Mark as initialized
        g_init = True
    
    safe_log("global init done")

def result_visualization(img, bbox):
    """
    Visualize detection/tracking results on the input image and display it.
    """
    # Make a copy of input image
    oframe = img.copy()
    
    # Resize based on resolution (different behavior for RGB vs IR)
    if img.shape[0] > 800:  # RGB case (downscale)
        # Use integer dimensions for resize
        new_width = int(oframe.shape[1] * 0.5)
        new_height = int(oframe.shape[0] * 0.5)
        magnification = 0.5  # Scale factor for coordinates
    else:  # IR case (keep original size)
        new_width = int(oframe.shape[1])
        new_height = int(oframe.shape[0])
        magnification = 1    # No scaling for coordinates
    
    # Resize the image for display
    visuframe = cv2.resize(oframe, (new_width, new_height), cv2.INTER_LINEAR)
    
    # Scale bounding box coordinates according to resize factor
    bbx=[int(i*magnification) for i in bbox]
    
    # Draw rectangle around detected/tracked object
    cv2.rectangle(visuframe,(bbx[0],bbx[1]), (bbx[0]+bbx[2],bbx[1]+bbx[3]), (0,255,0), 2)
    
    # Show the visualization
    cv2.imshow("tracking", visuframe)
    cv2.waitKey(1)  # Update display with 1ms delay

def imgproc(data, current_video_path=""):
    """
    Process a frame for drone detection and tracking.
    
    Args:
        data: Input frame data
        current_video_path: Path to the current video file (optional)
    """
    global g_init, g_detector, g_tracker, g_frame_counter, count
    global detect_box, track_box, repeat_detect, detect_first, IMG_TYPE
    
    # Use default video path if not provided
    if not current_video_path and 'video_path' in globals():
        current_video_path = video_path
    
    # Update frame counter and log
    count += 1
    safe_log('recv a frame')
    bbx = None

    # Convert input to numpy array
    frame = np.array(data)
    #print('frame.shape=',frame.shape)
    
    # Automatic video type detection
    IMG_TYPE = 0  # Default: RGB
    if len(frame.shape) == 2:  # Grayscale
        IMG_TYPE = 1
        frame = mono_to_rgb(frame)
        #print("Detected: Grayscale -> IR")
    elif len(frame.shape) == 3:  # RGB format
        # If filename starts with 'n' (night vision) or contains 'ir', process as IR
        filename = os.path.basename(current_video_path).lower() if current_video_path else ""
        if filename.startswith('n') or 'ir' in current_video_path.lower():
            IMG_TYPE = 1
            #print(f"Detected: {filename} -> IR (night vision)")
        # Or if all channels are identical, it's grayscale in RGB format
        elif np.array_equal(frame[:,:,0], frame[:,:,1]) and np.array_equal(frame[:,:,1], frame[:,:,2]):
            IMG_TYPE = 1
            #print("Detected: Grayscale RGB -> IR")
        else:
            IMG_TYPE = 0
            #print("Detected: True RGB")
    
    #print(f"Video tipi: {'IR' if IMG_TYPE == 1 else 'RGB'}, Frame: {frame.shape}")

    if g_detector and g_tracker:
        #print("1")
        # g_frame_counter = 0
        if g_frame_counter <= 0:
            if IMG_TYPE == 0:
                safe_log('RGB Mode: {}'.format(IMG_TYPE))
                init_box = g_detector.forward_RGB(frame)
                center_box = [320,192,0,0]  # RGB için merkez
                #print(f"RGB Merkezi: {center_box}")
            else:
                safe_log('IR Mode: {}'.format(IMG_TYPE))
                init_box = g_detector.forward_IR(frame)
                center_box = [320,256,0,0]  
            
            #print("2")
            #print(count)
            # First detection mode
            if detect_first:
                # If a detection is found
                if init_box is not None:
                    # Check if detection is close to center and timing is right
                    if distance_check(init_box, center_box, 60) and count % 4 == 1:
                        # Initialize tracker with detection
                        g_tracker.init_track(init_box, frame)
                        g_frame_counter = TRACK_MAX_COUNT
                        safe_log('init done')
                        detect_first = False  # Switch to tracking mode
                    else:
                        # Detection is too far from center or timing isn't right
                        pass
                        
                    # Convert detection coordinates to integers
                    init_box = [int(x) for x in init_box]
                    
                    # Send coordinates periodically based on image type
                    if IMG_TYPE == 1 and count % 8 == 1:  # IR mode
                        if sendLocation:
                            send_coord(init_box)
                        count = 1  # Reset counter
                    elif IMG_TYPE == 0 and count % 8 == 1:  # RGB mode
                        # Scale coordinates from detection to display resolution
                        init_box = scale_coords([640,384], init_box, [1920,1080])
                        if sendLocation:
                            send_coord(init_box)
                        count = 1  # Reset counter
                        
                    # Visualize results if enabled
                    if Visualization:
                        result_visualization(frame, init_box)
                # Reset counter if we've been searching too long without detection
                elif count <= 1 or count >= 9:
                    count = 0
            else:
                # If a detection is found and it's close enough to center,
                # update tracker with the new detection
                # If a detection is found and it's close enough to center,
                # update tracker with the new detection
                if init_box is not None and distance_check(init_box, center_box,60):
                    g_tracker.change_state(init_box)  # Update tracker with new detection
                    g_frame_counter = TRACK_MAX_COUNT  # Reset tracking counter
                    init_box = [int(x) for x in init_box]  # Convert to integers
                    
                    # Send coordinates if needed (different handling for IR vs RGB)
                    if IMG_TYPE ==1 and count % 2 == 1:  # IR mode
                        if sendLocation: 
                            send_coord(init_box)
                    elif IMG_TYPE == 0 and count % 2 == 1:  # RGB mode
                        # Scale coordinates from detection resolution to display resolution
                        init_box = scale_coords([640,384], init_box, [1920,1080])
                        if sendLocation:
                            send_coord(init_box)
                            
                    # Visualize results if enabled
                    if Visualization:
                        result_visualization(frame, init_box)
                else:
                    # Continue tracking existing object
                    g_frame_counter = TRACK_MAX_COUNT
                    bbx = g_tracker.on_track(frame)  # Get tracking results
                    g_frame_counter -= 1
                    
                    # Log tracking results
                    send_bbs('get!--{}'.format(bbx))
                    
                    # Send coordinates if needed (different handling for IR vs RGB)
                    if IMG_TYPE ==1 and count % 2 == 1:  # IR mode
                        if sendLocation:
                            send_coord(bbx)
                    elif IMG_TYPE == 0 and count % 2 == 1:  # RGB mode
                        # Scale coordinates from detection resolution to display resolution
                        bbx = scale_coords([640,384], bbx, [1920,1080])
                        if sendLocation:
                            send_coord(bbx)
                            
                    # Visualize results if enabled
                    if Visualization:
                        result_visualization(frame, bbx)
                
        else:
            # Continue tracking without new detection
            bbx = g_tracker.on_track(frame)  # Get tracking results
            
            # Decrement tracking counter
            g_frame_counter -= 1
            
            # Log tracking results
            send_bbs('get!--{}'.format(bbx))
            
            # Send coordinates if needed (different handling for IR vs RGB)
            if IMG_TYPE ==1 and count % 2 == 1:  # IR mode
                if sendLocation:              
                    send_coord(bbx)
            elif IMG_TYPE == 0 and count % 2 == 1:  # RGB mode
                # Scale coordinates from detection resolution to display resolution
                bbx = scale_coords([640,384], bbx, [1920,1080])
                if sendLocation:
                    send_coord(bbx)
                    
            # Visualize results if enabled
            if Visualization:
                result_visualization(frame, bbx)

            

        
def getUDPSocket(IpAddr, Port):
    """
    Create and bind a UDP socket to the specified IP address and port.
    """
    server = socket.socket(type=socket.SOCK_DGRAM)  # Create UDP socket
    server.bind((IpAddr, Port))                     # Bind to address
    safe_log("Server Socket is READY!")
    return server
    
def udpRecv(server, frameSize):
    """
    Receive image data via UDP socket using a custom protocol.
    
    Args:
        server: UDP socket server that's already bound and listening
        frameSize: Size of frame chunks to receive
        
    Returns:
        numpy.ndarray: Reconstructed image as a numpy array, either grayscale or RGB
    """
    global g_data
    img_recv_final = None
    
    # Receive initial marker
    g_data, addr = server.recvfrom(256)  # Should be b'I BEGIN'
    
    # Process data if the correct marker is received
    if g_data == b'I BEGIN':
        print("I BEGIN!!")
        server.sendto(b'sucess', addr)  # Acknowledge receipt
        
        # Receive image size
        data, addr = server.recvfrom(16)  # Size of the image to be sent
        server.sendto(b'sucess', addr)    # Acknowledge receipt
        img_size = int.from_bytes(data, byteorder='little')  # Convert bytes to integer
        
        # Receive image data in chunks
        img_recv_all = b''
        recvd_size = 0
        
        while recvd_size < img_size:
            data, addr = server.recvfrom(frameSize)  # Wait for data chunk
            server.sendto(b'sucess', addr)           # Acknowledge receipt
            
            img_recv_all += data          # Append data
            recvd_size += frameSize        # Update received size
        
        # Convert received bytes to numpy array
        img_recv_final = np.fromstring(img_recv_all, np.uint8)
        
        # Reshape based on image size (grayscale vs. RGB)
        if img_size < 500000:  # Smaller images are grayscale
            img_recv_final = img_recv_final.reshape((512, 640))
        else:  # Larger images are RGB
            img_recv_final = img_recv_final.reshape((384, 640, 3))
            
    return img_recv_final


if __name__== "__main__": 
    """
    Main execution block for drone detection and tracking system.
    """
    # Initialize global variables and models
    global_init()
    
    # UDP socket configuration
    addr = '127.0.0.1'  # Local loopback address
    port = 9999         # Port number
    frameSize = 8192    # UDP packet size
    detect_num = 5      # Number of detections before tracking
    
    # Create and bind UDP socket
    server = getUDPSocket(addr, port)
    
    # Open test video file
    video_path = r"C:\Users\kinac\DroneDetection\Anti-UAV\Codes\testvideo\testvideo\n2.mp4"
    cap = cv2.VideoCapture(video_path)
    
    # Print startup information
    print("Start!!")
    print("path:", video_path)
    print("exists:", os.path.exists(video_path))
    print("opened:", cap.isOpened())

    # Main processing loop
    while True:
        # Read a frame from video
        # Alternative method: data = udpRecv(server, frameSize)
        ret, data = cap.read()

        # Check for stop signal
        if g_data == b'STOP':
            g_frame_counter = 0
            
        # Check if video has ended
        if not ret:
            print("Video ended, exiting.")
            cap.release()
            cv2.destroyAllWindows()
            exit()  # Exit the program when the video ends

        # Process frame if available
        if data is not None:
            imgproc(data, video_path)  # Process with detection and tracking
