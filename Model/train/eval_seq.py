import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def preprocess_and_analyze_image(input_image_path, output_folder):
    """
    Preprocess the image and analyze the contours in it.
    Operations include pixel conversion, contour finding, filtering and drawing.
    """
    img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)

    img[np.where((img[:, :, 0]!= 255) | (img[:, :, 1]!= 255) | (img[:, :, 2]!= 255))] = [0, 0, 1]
    img[np.where((img[:, :, 0] == 255) & (img[:, :, 1] == 255) & (img[:, :, 2] == 255))] = [0, 0, 0]
    img[np.where((img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 1))] = [255, 255, 255]

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    exterior_contour = []
    for idx, hiearchy in enumerate(hierarchy[0]):
        if hiearchy[-1] == -1:
            exterior_contour.append(contours[idx])
    final_img = cv2.drawContours(img, exterior_contour, -1, (255, 0, 0), 2)


    cv2.imwrite(output_folder + os.path.splitext(os.path.basename(input_image_path))[0] + '.png', final_img)


    return len(exterior_contour)

def save_data_to_txt(file_path, data_dict):
    """
    Save the data dictionary to a text file.
    Assume the file path is correct.
    """
    with open(file_path, 'w') as f:
        for key, value in data_dict.items():
            f.write(f"{key}: {value}\n")

def main():
    parser = argparse.ArgumentParser(description='Image processing and data analysis tool')

    parser.add_argument('--novel', type=str, required=True,help='Path to the novel pair text file')
    parser.add_argument('--unique', type=str, required=True,help='Path to the unique pair text file')
    parser.add_argument('--data_folder', type=str, required=True,help='All CAD images')
    parser.add_argument('--output_folder', type=str, required=True,help='Folder to save output images')
    parser.add_argument('--coutpair_file', type=str, required=True,help='Path to the count pair summary file')
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    novelcount = 0
    unqiuecount = 0
    novelpair = []
    unqiuepair = []
    count = 0
    countpair = []

    with open(args.novel, 'r') as f1:
        novellines = [line.strip() for line in f1.readlines()]

    with open(args.unique, 'r') as f2:
        uniquelines = [line.strip() for line in f2.readlines()]

    images = os.listdir(args.data_folder)

    print("Processing images and analyzing contours...")
    for image in tqdm(images):
        numstr = image.split('_')[0]
        numcontour = preprocess_and_analyze_image(args.data_folder + image, args.output_folder)
        if numcontour > 1:
            count += 1
            countpair.append(str(int(numstr)))
            if str(int(numstr)) in novellines:
                novelcount += 1
                novelpair.append(str(int(numstr)))
            if str(int(numstr)) in uniquelines:
                unqiuecount += 1
                unqiuepair.append(str(int(numstr)))

    print("Counting unused entries in text documents...")
    numstr = [str(int(image.split('_')[0])) for image in images]
    nouse_novelcount = 0
    nouse_unqiuecount = 0
    nouse_novelpair = []
    nouse_unqiuepair = []

    for line in tqdm(novellines):
        if line not in numstr:
            nouse_novelcount += 1
            nouse_novelpair.append(line)

    for line in tqdm(uniquelines):
        if line not in numstr:
            nouse_unqiuecount += 1
            nouse_unqiuepair.append(line)

    try:
        with open(args.coutpair_file, 'r') as f:
            content = f.read()
            novel_number = int(content.split('NovelNumber: ')[1].split('\n')[0].strip('[]'))
            unique_number = int(content.split('UniqueNumber: ')[1].split('\n')[0].strip('[]'))
    except:
        print("Failed to read the statistics file, using the number of processed images as the default value")
        novel_number = count
        unique_number = count

    tqdmnumber = len(images)
    new_novel = (novel_number - nouse_novelcount - novelcount) / tqdmnumber if tqdmnumber > 0 else 0
    new_unqiue = (unique_number - nouse_unqiuecount - unqiuecount) / tqdmnumber if tqdmnumber > 0 else 0
    print(f"novel=({novel_number}-{nouse_novelcount}-{novelcount})/{tqdmnumber}={new_novel}")
    print(f"unqiue=({unique_number}-{nouse_unqiuecount}-{unqiuecount})/{tqdmnumber}={new_unqiue}")
