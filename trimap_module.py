#!/usr/bin/env python
import argparse

import cv2, os, sys
import numpy as np
import tqdm


def extractImage(path):
    # error handller if the intended path is not found
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return image


def checkImage(image):
    """
    Args:
        image: input image to be checked
    Returns:
        binary image
    Raises:
        RGB image, grayscale image, all-black, and all-white image

    """
    if len(image.shape) > 2:
        print("ERROR: non-binary image (RGB)");
        sys.exit();

    smallest = image.min(axis=0).min(axis=0)  # lowest pixel value: 0 (black)
    largest = image.max(axis=0).max(axis=0)  # highest pixel value: 1 (white)

    if (smallest == 0 and largest == 0):
        print("ERROR: non-binary image (all black)");
        sys.exit()
    elif (smallest == 255 and largest == 255):
        print("ERROR: non-binary image (all white)");
        sys.exit()
    elif (smallest > 0 or largest < 255):
        print("ERROR: non-binary image (grayscale)");
        sys.exit()
    else:
        return True


class Toolbox:
    def __init__(self, image):
        self.image = image

    @property
    def printImage(self):
        """
        Print image into a file for checking purpose
        unitTest = Toolbox(image);
        unitTest.printImage(image);
        """
        f = open("image_results.dat", "w+")
        for i in range(0, self.image.shape[0]):
            for j in range(0, self.image.shape[1]):
                f.write("%d " % self.image[i, j])
            f.write("\n")
        f.close()

    @property
    def displayImage(self):
        """
        Display the image on a window
        Press any key to exit
        """
        cv2.imshow('Displayed Image', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def saveImage(self, title, extension):
        """
        Save as a specific image format (bmp, png, or jpeg)
        """
        cv2.imwrite("{}.{}".format(title, extension), self.image)

    def morph_open(self, image, kernel):
        """
        Remove all white noises or speckles outside images
        Need to tune the kernel size
        Instruction:
        unit01 = Toolbox(image);
        kernel = np.ones( (9,9), np.uint8 );
        morph  = unit01.morph_open(input_image, kernel);
        """
        bin_open = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)
        return bin_open

    def morph_close(self, image, kernel):
        """
        Remove all black noises or speckles inside images
        Need to tune the kernel size
        Instruction:
        unit01 = Toolbox(image);
        kernel = np.ones( (11,11)_, np.uint8 );
        morph  = unit01.morph_close(input_image, kernel);
        """
        bin_close = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)
        return bin_close


def trimap(image, name, size, erosion=False, save_path="./images/results/"):
    """
    This function creates a trimap based on simple dilation algorithm
    Inputs [4]: a binary image (black & white only), name of the image, dilation pixels
                the last argument is optional; i.e., how many iterations will the image get eroded
    Output    : a trimap
    """
    checkImage(image)
    pixels = 2 * size + 1  ## Double and plus 1 to have an odd-sized kernel
    kernel = np.ones((pixels, pixels), np.uint8)  ## Pixel of extension I get

    if erosion is not False:
        erosion = int(erosion)
        erosion_kernel = np.ones((3, 3), np.uint8)  ## Design an odd-sized erosion kernel
        image = cv2.erode(image, erosion_kernel, iterations=erosion)  ## How many erosion do you expect
        image = np.where(image > 0, 255, image)  ## Any gray-clored pixel becomes white (smoothing)
        # Error-handler to prevent entire foreground annihilation
        if cv2.countNonZero(image) == 0:
            print("ERROR: foreground has been entirely eroded")
            sys.exit()

    dilation = cv2.dilate(image, kernel, iterations=1)
    dilation[dilation == 255] = 127
    dilation[image > 127] = 200
    dilation[dilation < 127] = 0
    dilation[dilation > 200] = 0
    dilation[dilation == 200] = 255
    dilation[(dilation != 0) & (dilation != 255)] = 127

    ## Change the directory
    new_name = '{}px_'.format(size) + name + '.png'
    image_path = os.path.join(save_path, new_name)
    if os.path.exists(image_path):
        if image_path[-5].isdigit():
            num = int(image_path[-5]) + 1
        else:
            num = 1
        new_name = '{}px_'.format(size) + name + "_" + str(num) + '.png'
        image_path = os.path.join(save_path, new_name)
    cv2.imwrite(image_path, dilation)
    print(image_path)


def findBox(image):
    image_w = image.sum(0)
    res_w = np.where(image_w > 0)
    x1, x2 = res_w[0][0], res_w[0][-1]
    image_h = image.sum(1)
    res_h = np.where(image_h > 0)
    y1, y2 = res_h[0][0], res_h[0][-1]
    return x1, x2, y1, y2


def dilate_and_erode(mask_data, struc="ELLIPSE", size=(10, 10)):
    """
    膨胀侵蚀作用，得到粗略的trimap图
    :param mask_data: 读取的mask图数据
    :param struc: 结构方式
    :param size: 核大小
    :return:
    """
    if struc == "RECT":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    elif struc == "CORSS":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, size)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)

    msk = mask_data / 255

    dilated = cv2.dilate(msk, kernel, iterations=1) * 255
    eroded = cv2.erode(msk, kernel, iterations=1) * 255
    res = dilated.copy()
    res[((dilated == 255) & (eroded == 0))] = 128
    return res


def set_args():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Trimap generate")
    parser.add_argument("--images-dir", type=str, default="./images/masks")
    parser.add_argument("--save-dir", type=str, default='./images/results/')
    parser.add_argument("--single-image", action="store_true")
    parser.add_argument("--image-path", type=str, default="./images/practices/binary_0.png")

    parser.add_argument("--dilated-size", type=int, default=5)
    parser.add_argument("--dilated-epoch", type=int, default=4)

    return parser.parse_args()


def gen_trimap(img_path, epoch):
    img = extractImage(img_path)
    for i in range(epoch):
        unit01 = Toolbox(img)
        kernel1 = np.ones((11, 11), np.uint8)
        image = unit01.morph_close(img, kernel1)
        checkImage(image)
        # Double and plus 1 to have an odd-sized kernel
        pixels = 2 * size + 1
        # Pixel of extension I get
        kernel = np.ones((pixels, pixels), np.uint8)

        dilation = cv2.dilate(image, kernel, iterations=1)
        dilation[dilation == 255] = 127
        dilation[image > 127] = 200
        dilation[dilation < 127] = 0
        dilation[dilation > 200] = 0
        dilation[dilation == 200] = 255
        dilation[(dilation != 0) & (dilation != 255)] = 127

        img = dilation

    return img


if __name__ == '__main__':
    args = set_args()
    if args.single_image:
        images = [args.image_path]
    else:
        if os.path.exists(args.images_dir):
            images = [os.path.join(args.images_dir, p) for p in os.listdir(args.images_dir)]
        else:
            print(args.images_dir + "is not exists...")
            exit(0)
    save_path = args.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    size = args.dilated_size

    for img_path in tqdm.tqdm(images):
        img_name = img_path.split("/")[-1]
        if "\\" in img_name:
            img_name = img_name.split("\\")[-1]
        trimap = gen_trimap(img_path, args.dilated_epoch)
        cv2.imwrite(os.path.join(save_path, img_name), trimap)
