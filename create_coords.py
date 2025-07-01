import openslide
from openslide import ImageSlide, open_slide
from openslide.deepzoom import DeepZoomGenerator

import argparse
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import cv2
import json
import os

from rl_benchmarks.constants import UNN_PATHS, TCGA_PATHS, S36_PATHS, NLST_PATHS

"""

Input:
  source: json file w paths to all wsi files.
  patch size, magnification, percentage, save_dir.
Output:
  files are saved in save_dir (one file for each wsi), containing coordinates for patches. 
  Stored as (tile_level, x_coordinate, y_coordinate)
"""

def process_slide(slide_path, args):
    slide_name = slide_path.split("/")[-1]

    # Move on if coordinates for this .svs file is already created!
    dir_name = args.save_dir+"/"+slide_name
    if Path(dir_name).exists():
        print(f"  Coordinates for {slide_name} already exists and can be found at {dir_name}")
        return

    print(f"  Extract tile coordinates for {slide_name}", flush=True)

    patch_size = args.patch_size
    percentage = args.percentage
    mthresh = 7
    seg_level = -1
    sthresh = 8
    sthresh_up = 255
    use_otsu = True
    close = 4 # Actually 0

    try:
        slide = open_slide(slide_path)
    except:
        p1 = "/".join(slide_path.split("/")[:-2])
        p2 = "-".join(slide_path.split("/")[-2].split("-")[:-1])
        p3 = slide_path.split("/")[-1]
        slide_path_new = p1+"/"+p2+"/"+p3
        print(f"   Slide {slide_path} could not be opened. Try {slide_path_new} instead.")
        slide_path = slide_path_new
        slide = open_slide(slide_path)
        
    tiles = DeepZoomGenerator(slide, tile_size=patch_size, overlap=0, limit_bounds=True)

    #print("The number of levels in the tiles object are: ", tiles.level_count) # eg 17
    #print("The dimensions of data in each level are: ", tiles.level_dimensions)
    #print("Total number of tiles = : ", tiles.tile_count)
    #How many tiles at a specific level?
    #if magnification == 20:
    #elif magnification == 40:
    if "aperio.MPP" in slide.properties.keys():
        #mpp of the slide is available from aperio
        mpp = float(slide.properties["aperio.MPP"])
    elif "openslide.mpp-x" in slide.properties.keys() and "openslide.mpp-y" in slide.properties.keys():
        mpp = float(slide.properties["openslide.mpp-x"]+slide.properties["openslide.mpp-y"])/2
        # Use average (they are typically the same)
    else:
        print(f"\n\nUnknown mpp for slide {slide_name}. If non-aperio scanner slide, please add functionality for this. Stopping the process of coordinate creation.", flush=True)
        print(f"Coordinates for slide {slide_name} not created!")
        # import IPython
        # IPython.embed()
        # import sys
        # sys.exit()

    if mpp-0.25 > 0.125: # magnification is around 20x.
        level_num = tiles.level_count-1
    else: # aperio.MPP is closer to 0.25 than 0.5, so magnification is around 40x.
        level_num = tiles.level_count-2
    print(f"  Level {level_num}", flush=True)

    # Use Otsu to find slide foreground / background
    from PIL import Image, ImageOps
    seg_dim = slide.level_dimensions[-1]
    if seg_level == -1:
        seg_level = len(slide.level_downsamples) - 1
    img = slide.read_region((0,0), seg_level, seg_dim)
    #img_hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)  # Convert to HSV space
    img_g = np.array(ImageOps.grayscale(img))
    #img_g = 255-img_g
    #print("min max", img_g.min(), img_g.max())
    img_med = cv2.medianBlur(img_g, mthresh)  # Apply median blurring
    #img_med = cv2.medianBlur(img_hsv[:,:,1], mthresh)  # Apply median blurring

    
    #img_g.save(f"slides/gray/{slide_path.split('/')[-1].split('_')[0]}.png")
    #return
    #sx = cv2.convertScaleAbs(cv2.Sobel(np.array(img_g), cv2.CV_64F, 1, 0, ksize=3))
    #sy = cv2.convertScaleAbs(cv2.Sobel(np.array(img_g), cv2.CV_64F, 0, 1, ksize=3))
    #img_edges = cv2.addWeighted(sx,0.5, sy,0.5, 0)
    #plt.imsave(f"edges/{slide_path.split('/')[-1].split('_')[0]}.png", img_edges, cmap='gray')
    #return

    #img_rgb = img.convert('RGB')
    #img_rgb.save(f"slides/{slide_path.split('/')[-1].split('_')[0]}.png")
    #return
    # Thresholding
    if use_otsu:
        thres, img_otsu = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    else:
        thres, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)
    # Morphological closing
    # if close > 0:
    #     kernel = np.ones((close, close), np.uint8)
    #     img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)                 

    #Image.fromarray(img_otsu).save(f"otsu_slides/gray/{slide_path.split('/')[-1].split('_')[0]}.png")
    #return

    cols, rows = tiles.level_tiles[level_num]
    coords = np.zeros((cols*rows,3))
    used = 0
    old_use = 0
    for row in range(rows):
        # if used>10:
        #     break
        for col in range(cols):
            temp_tile = tiles.get_tile(level_num, (col, row))
            temp_tile_RGB = temp_tile.convert('RGB')
            # print(type(temp_tile), type(temp_tile_RGB))
            # temp_tile_np = np.array(temp_tile_RGB)
            # if temp_tile_np.shape != (patch_size,patch_size,3):
            #     continue

            gray_tile = ImageOps.grayscale(temp_tile)
            gray_tile_np = np.array(gray_tile)
            #print("Have gray tile")
            if gray_tile_np.shape != (patch_size,patch_size):
                continue
            thres_tile = gray_tile_np < thres
            perc_tissue = np.sum(thres_tile)/(patch_size*patch_size)
            if perc_tissue > percentage:
                old_use += 1
                col_var = np.std(gray_tile_np)
                if col_var > 10:    # Sanity check that the tile is not homogenous in intensity.
                    #print("yes", col_var, row, col)
                    #temp_tile_RGB.save(f"keep_tiles_2/{slide_path.split('/')[-1].split('_')[0]}_{row}_{col}.png")
                    coords[used,:] = level_num,col,row
                    used += 1
            # if used > 10:
            #     break

            # # Check how much tissue we have
            # tile_hsv = cv2.cvtColor(temp_tile_np, cv2.COLOR_RGB2HSV)  # Convert to HSV space
            # thres_tile = tile_hsv[:,:,1] > thres
            # perc_tissue = np.sum(thres_tile)/(patch_size*patch_size)
            # if perc_tissue > percentage:
            #     #print(type(temp_tile_RGB))
            #     # Check that we have some edges as well
            #     gray_tile = ImageOps.grayscale(temp_tile_RGB)
            #     sx = cv2.convertScaleAbs(cv2.Sobel(np.array(gray_tile), cv2.CV_64F, 1, 0, ksize=3))
            #     sy = cv2.convertScaleAbs(cv2.Sobel(np.array(gray_tile), cv2.CV_64F, 0, 1, ksize=3))
            #     edges_tile = cv2.addWeighted(sx,0.5, sy,0.5, 0)
            #     edge_nr = edges_tile.mean()
            #     old_use += 1

            #     if edge_nr > 40:
            #         coords[used,:] = level_num,col,row
            #         used += 1
            #         #print("edge_nr", edge_nr, row, col)

            #         #temp_tile_RGB.save(f"keep_tiles_2/{slide_path.split('/')[-1].split('_')[0]}_{row}_{col}.png")
        
    print(f"Used {used} tiles. Prev alg would have used {old_use} tiles.")
    
    # Delete unused rows
    coords = coords[:used,:]
    dir_name = args.save_dir+"/"+slide_name
    os.makedirs(dir_name, exist_ok=True)
    np.save(dir_name+"/coords.npy", coords)
    print(f"  Extracted {coords.shape[0]} tiles out of {cols*rows} possible:   {slide_name}", flush=True)


def main(args):
    with open("data/"+args.source) as f:
        data = json.load(f)

    # d_20 = data["20"]
    # d_40 = data["40"]
    # slides = d_20+d_40

    slides = data["Relative_paths"]
    # # Which files to process
    # if args.magnification==20:
    #     slides=d_20
    # elif args.magnification==40:
    #     slides=d_40
    # else:
    #     print("We only have slides w magnification 20 or 40. Check argparse args")
    #     import sys
    #     sys.exit()

    slides = sorted(slides)
    # slides = slides[300:]
    # print("Process slides 300 and out")

    if args.dataset.upper()=="TCGA":
        data_path = TCGA_PATHS["SLIDES"]
    elif args.dataset.upper()=="UNN":
        data_path = UNN_PATHS["SLIDES"]
    elif args.dataset.upper()=="S36":
        data_path = S36_PATHS["SLIDES"]
    elif args.dataset.upper()=="NLST":
        data_path = NLST_PATHS["SLIDES"]
    else:
        print("This dataset is not implemented in create_coords yet.")

    if args.i == -1:
        start=0
        stop = len(slides)
    else:
        start=args.i
        stop=np.min([args.i+300, len(slides)])

    for i in range(start, stop): #len(slides)):
        slide = slides[i]
        print(f"{i}/{len(slides)}: {slide.split('/')[-1]}", flush=True)
        try:
            process_slide(data_path+slide, args)
        except Exception as e:
            print(f"   Slide {slide.split('/')[-1]} could not be processed. Exception {e}\n\n", flush=True)
        i += 1



print("Starting create_coords")
parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type = str, default='as_hipt.json',
					help='path to json containing paths to all wsi image files')
parser.add_argument('--dataset', type=str, default="TCGA", help="Which dataset to work on")
parser.add_argument('--patch_size', type = int, default=256,
					help='patch_size')
#parser.add_argument('--magnification', type=int, default=20, help='Deprecated. Which magnification of images to process.')
parser.add_argument('--percentage', default=0.6, type=float, help='How many percent of tile needs to be tissue to keep the tile')
parser.add_argument('--save_dir', type = str,
					help='directory to save processed data')
parser.add_argument('--i', type=int, default=-1, help='Index to start processing')


if __name__ == '__main__':
    args = parser.parse_args()
    print(args, flush=True)
    main(args)
    print("Finish create")