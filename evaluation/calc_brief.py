import cv2
import copyreg

def _pickle_keypoint(keypoint): #  : cv2.KeyPoint
    return cv2.KeyPoint, (
        keypoint.pt[0],
        keypoint.pt[1],
        keypoint.size,
        keypoint.angle,
        keypoint.response,
        keypoint.octave,
        keypoint.class_id,
    )
# Apply the bundling to pickle
copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoint)

######################################################## model setup & download
def precompute(filename):
    img = cv2.imread(filename)
    if img is None:
        print("could not open img:", filename)
        return {}
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # as recommended in opencv
    # Initiate FAST detector
    star = cv2.xfeatures2d.StarDetector_create()
    # Initiate BRIEF extractor
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    # find the keypoints with STAR
    kp = star.detect(img,None)
    # compute the descriptors with BRIEF
    kp, des = brief.compute(gray, kp)
    return {
        "keypoints": kp,
        "descriptors": des
    }