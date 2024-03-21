import os
import argparse

import numpy as np
from PIL import Image
from rembg import remove

import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from segment import Segmenter
from inpaint import inPainter
from view import generate_view
from utils import saveImage, threshold_mask, dilate_and_erode, move_image, shift_object, highlight_object


argparser = argparse.ArgumentParser()

argparser.add_argument('-i','--image', help='Path to the image', default='images/img.jpg')
argparser.add_argument('-c', '--class_', help='Class to be identified', default='chair')
argparser.add_argument('-o', '--output', help='Output path', default='outputs/')
argparser.add_argument('-tk', '--task', help='Task to be performed', default='2')
argparser.add_argument('-az', '--azimuth', help='Azimuth angle', default=0)
argparser.add_argument('-po', '--polar', help='Polar angle', default=0)

args = argparser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    os.mkdir(args.output + args.class_)
except:
    print("All Set")

if args.task == '1':
    pipeLine = Segmenter(DEVICE)
    image = Image.open(args.image)
    print("Segmenting the image...")
    mask, _ = pipeLine(image.copy(), args.class_)
    mask = np.array(mask*255).astype(np.uint8)
    mask = threshold_mask(mask)
    mask = dilate_and_erode(mask, 7, 2)

    finalImg = highlight_object(np.array(image), mask)
    saveImage(finalImg, args.output + args.class_ + '/' + 'highlighted.jpg')

    print("Mask generated successfully!")

elif args.task == '2':
    pipeLine = Segmenter(DEVICE)
    image = Image.open(args.image)

    print("Segmenting the image...")
    mask, leftCornerOg = pipeLine(image.copy(), args.class_)
    leftCornerOg = leftCornerOg[0:2]
    mask = np.array(mask*255).astype(np.uint8)
    mask = threshold_mask(mask)
    mask = dilate_and_erode(mask, 7, 2)
    saveImage(mask, args.output + args.class_ + '/' + 'mask.jpg')
    mask_ = mask.copy()
    print("Mask generated successfully!")

    print("Extracting the object from the image...")
    object2D = np.array(image)
    object2D = object2D.astype(np.float32) / 255.0
    mask = mask.astype(np.float32) / 255.0
    object2D = object2D * mask[:,:,None] + (1 - mask)[:,:,None]
    object2D = (object2D * 255.0).astype(np.uint8)
    object2D = move_image(object2D, mask)
    object2D = Image.fromarray(object2D)
    object2D.save(args.output + args.class_ + '/' + 'object.jpg')
    print("Object extracted successfully!")

    print("Inpainting rest of the image...")
    mask = mask_
    inPaintPipe = inPainter(DEVICE)
    background = inPaintPipe(image, mask.copy(), args.class_)
    background.save(args.output + args.class_ + '/' + 'background.jpg')
    del inPaintPipe
    print("Inpainting done successfully!")

    print("Generating Novel View...")
    image = Image.open(args.image)
    object2D = Image.open(args.output + args.class_ + '/' + 'object.jpg')
    background = Image.open(args.output + args.class_ + '/' + 'background.jpg')
    background = np.array(background.resize(image.size))
    
    xs = float(args.polar)
    ys = float(args.azimuth)

    view = generate_view(object2D, xs, ys, DEVICE).resize(image.size)
    
    mask, leftcorner = pipeLine(view, args.class_)
    leftcorner = leftcorner[:2]
    
    view = np.array(view)
    view, mask = shift_object(view, mask, leftcorner, leftCornerOg)
    mask = mask / np.max(mask)
    
    saveImage(view, args.output + args.class_ + "/" + "novel_view.jpg")
    mask_inv = 1 - mask
    foreground = view.copy()
    finalImg = background * mask_inv[:,:,None] + foreground * mask[:,:,None]
    finalImg = Image.fromarray(finalImg.astype(np.uint8))
    finalImg.save(args.output + args.class_ + '/' + 'finalImg.jpg')

    print("Novel View generated successfully!")
