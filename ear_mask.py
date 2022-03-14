from importlib.resources import path
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
import torchvision
import cv2
import numpy as np
import math
from PIL import Image

## class to import and execute onnx runtime for ead mask

class Mask_RCNN():

    def __init__(self, path_to_onnx):
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,256,3)
        self.model.load_state_dict(torch.load('./model_detection/ear_segmentation.onnx'))
        self.model.eval()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    def __call__(self, image):
        to_tensor = transforms.ToTensor()
        tensor_img = to_tensor(image)
        prediction = self.model([tensor_img.to(self.device)])
        boxes = prediction[0]
        label = prediction[1]
        label_perc = prediction[2]
        segmentation = prediction[3]
        return (boxes, label, label_perc, segmentation)

## helper function to transform segmentation matrix to image
def segmentation_to_image(segmentation):
    msk = segmentation*255
    msk = msk.astype(np.uint8)
    return msk


## Start helper function for ear normalization

def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in filter(lambda x: (x.shape[0] != 0), im_list)]
    return cv2.vconcat(im_list_resize)

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in filter(lambda x: (x.shape[1] != 0), im_list)]
    return cv2.hconcat(im_list_resize)

def rotate(rotateImage, angle):
    # Taking image height and width
    (imgHeight, imgWidth) = rotateImage.shape[:2]
    
    # Computing the centre x,y coordinates
    # of an image
    centreY, centreX = imgHeight//2, imgWidth//2
  
    # Computing 2D rotation Matrix to rotate an image
    rotationMatrix = cv2.getRotationMatrix2D((centreY, centreX), angle, 1.0)
  
    # Now will take out sin and cos values from rotationMatrix
    # Also used numpy absolute function to make positive value
    cosofRotationMatrix = np.abs(rotationMatrix[0][0])
    sinofRotationMatrix = np.abs(rotationMatrix[0][1])
  
    # Now will compute new height & width of
    # an image so that we can use it in
    # warpAffine function to prevent cropping of image sides
    diagonal = int(math.sqrt(imgWidth**2+imgHeight**2))
    newImageHeight = diagonal
    newImageWidth = diagonal
  
    # After computing the new height & width of an image
    # we also need to update the values of rotation matrix
    rotationMatrix[0][2] += (newImageWidth/2) - centreX
    rotationMatrix[1][2] += (newImageHeight/2) - centreY
  
    # Now, we will perform actual image rotation
    rotatingimage = cv2.warpAffine(
        rotateImage, rotationMatrix, (newImageWidth, newImageHeight))
    
    # get the nonzero alpha coordinates
    y,x,_ = rotatingimage.nonzero() 
    minx = np.min(x)
    miny = np.min(y)
    maxx = np.max(x)
    maxy = np.max(y) 
    cropImg = rotatingimage.copy()
    cropImg = cropImg[miny:maxy, minx:maxx]

    return cropImg

def draw_ellps_metadata(elps,color):
    (x, y), (MA, ma), angle = elps
    #draw ellypses
    color = cv2.ellipse(color, elps, (0,0,255), 2)
    #draw ellipses center
    color = cv2.circle(color, (int(x),int(y)), 4, (0, 0, 0), -1)
    # draw vertical line
    # compute major radius
    rmajor = max(MA,ma)/2
    if angle > 90:
        rmagangle = angle - 90
    else:
        rmagangle = angle + 90
    xtop = x + math.cos(math.radians(rmagangle))*rmajor
    ytop = y + math.sin(math.radians(rmagangle))*rmajor
    xbot = x + math.cos(math.radians(rmagangle+180))*rmajor
    ybot = y + math.sin(math.radians(rmagangle+180))*rmajor
    color = cv2.line(color, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (255, 255, 0), 2)
    return rmagangle, color

def get_angle_majaxe(elps):
    (x, y), (MA, ma), angle = elps
    # compute major radius
    rmajor = max(MA,ma)/2
    if angle > 90:
        rmagangle = angle - 90
    else:
        rmagangle = angle + 90
    return rmagangle

def get_final_mask(segmentation, label_perc, threshold):
    _,c,r = segmentation[0].shape
    final_mask = np.zeros((c, r, 1), dtype = "uint8")
    for inx in range(len(label_perc)):
        if label_perc[inx]<threshold:
            continue
        # show original image
        msk = segmentation_to_image(segmentation[inx][0])
        final_mask = cv2.bitwise_or(final_mask, msk)
    return final_mask


def get_ear_masked(path_input):
    model = Mask_RCNN("./model_detection/ear_segmentation.onnx")

    #path_input = f'./image/michele/m4.png'

    input_image_cv = cv2.imread(path_input)
    dimensions = input_image_cv.shape
    # check dimension for model input
    if dimensions != (702, 492, 3):
        return None

    input_image = Image.open(path_input)
    boxes, label, label_perc, segmentation = model(input_image)

    mask = get_final_mask(segmentation,label_perc, 0.60)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS )
    color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) 

    cc_mask = cv2.drawContours(color.copy(), contours, -1, (255,255,255), -1)
    kernel = np.ones((3,3), np.uint8)
    img_erosion = cv2.erode(cc_mask, kernel, iterations=10)
    img_erosion = cv2.cvtColor(img_erosion, cv2.COLOR_BGR2GRAY)
    final_mask = cv2.bitwise_or(mask, img_erosion)
    color = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR) 

    threshold_minarea = 0.05
    threshold_maxarea = 0.8

    min_area = (input_image_cv.shape[0]*input_image_cv.shape[1])*threshold_minarea
    max_area = (input_image_cv.shape[0]*input_image_cv.shape[1])*threshold_maxarea

    rmagangle = None
    if len(contours) != 0:
        for cont in contours:
            # for fit ellipses at leat 5 points 
            # arerequired
            if len(cont) < 5:
                continue
            elps = cv2.fitEllipse(cont)
            (x, y), (MA, ma), angle = elps
            A = 3.1415 / 4 * MA * ma
            if math.isnan(A):
                continue
            if A<min_area or A>max_area:
                continue
            rmagangle = get_angle_majaxe(elps)

    final_mask = cv2.GaussianBlur(final_mask,(21,21),0)
    masked_ear = cv2.bitwise_and(input_image_cv, input_image_cv, mask=final_mask)

    if rmagangle != None:
        output = rotate(masked_ear, rmagangle-90)
    else:
        output = masked_ear

    return output
    