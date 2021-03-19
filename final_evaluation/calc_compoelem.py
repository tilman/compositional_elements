import cv2
try:
    from compoelem.generate import global_action, pose_abstraction, pose_direction
    from compoelem.detect import converter
    from compoelem.detect.openpose_wrapper import get_poses
except ModuleNotFoundError:
    print("run as python module!!")
    exit()

# eval names:
# filterGac -> filter_pose_line_ga_result
# compare_pose_lines_2 -> cp2
# norm_by_global_action -> gacNorm
# minmax_norm_by_imgrect -> rectNorm

######################################################## config params
arch = "compoelem" #redo with fallback
eval_arch_name = "_"+arch+"_gacNorm_cpl2_filterGac"

######################################################## model setup & download
def precompute(filename):
    img = cv2.imread(filename)
    if img is None:
        print("could not open img:", filename)
        return {}
    img = converter.resize(img)
    width, height, _ = img.shape # type: ignore
    humans = get_poses(img)
    poses = converter.openpose_to_compoelem_poses(humans, *img.shape[:2]) # type: ignore
    pose_directions = pose_direction.get_pose_directions(poses)
    global_action_lines = global_action.get_global_action_lines(poses)
    pose_lines = pose_abstraction.get_pose_lines(poses)
    return {
        "humans": humans,
        "poses": poses,
        "pose_lines": pose_lines,
        #"pose_lines_fallback": pose_lines_fallback, # more robust poselines with fallback for maria
        "pose_directions": pose_directions,
        "global_action_lines": global_action_lines,
        "width": width,
        "height": height
    }