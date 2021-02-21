# import the necessary packages
import os
import cv2
import argparse
from imutils import paths


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
                help="path to input directory of images")
ap.add_argument("-t", "--threshold", type=float, default=100.0,
                help="focus measures that fall below this value will be considered 'blurry'")
args = vars(ap.parse_args())


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


# loop over the input images
for i, imagePath in enumerate(paths.list_images(args["images"])):
    # load the image, convert it to grayscale, and compute the
    # focus measure of the image using the Variance of Laplacian
    # method
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    print("laplacian calculated for frame #%s"%i)
    text = "Not Blurry"
    # if the focus measure is less than the supplied threshold,
    # then the image should be considered "blurry"
    if fm < args["threshold"]:
        text = "Blurry"
    # show the image
    cv2.putText(image, "{}: {:.2f}".format(text, fm), (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), thickness=5)
    # cv2.imshow("Image", image)
    # key = cv2.waitKey(0)
    cv2.imwrite("./blur_classified_frames/{0}_thresh_{1}_{2}.jpg".format(os.path.split(imagePath)[1], args['threshold'], i), image)
