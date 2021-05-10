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
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create(500)
    kp, des = surf.detectAndCompute(gray, None)
    return {
        "keypoints": kp,
        "descriptors": des
    }