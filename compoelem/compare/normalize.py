import cv2
from compoelem.types import *
from compoelem.config import config
from compoelem.detect import converter

def minmax_norm_point(point: Point, x_min:float, x_base:float, y_min:float, y_base:float) -> Point:
    x, y = point.xy
    x_new = (x[0]-x_min)/x_base
    y_new = (y[0]-y_min)/y_base
    return Point(x_new, y_new)

def minmax_norm(lines: Sequence[PoseLine], x_min:float, x_base:float, y_min:float, y_base:float) -> Sequence[PoseLine]:
    res_lines = []
    for line in lines:
        top_new = minmax_norm_point(line.top, x_min, x_base, y_min, y_base)
        bottom_new = minmax_norm_point(line.bottom, x_min, x_base, y_min, y_base)
        res_lines.append(PoseLine(top_new, bottom_new))
    return res_lines

def minmax_norm_by_imgrect(lines: Sequence[PoseLine], width, height) -> Sequence[PoseLine]:
    y_base, x_base, = height, width
    return minmax_norm(lines, 0, x_base, 0, y_base)

def minmax_norm_by_bbox(lines: Sequence[PoseLine]) -> Sequence[PoseLine]:
    if(len(lines) == 0):
        return lines
    point_cloud = []
    for line in lines:
        point_cloud.append(line.top.xy)
        point_cloud.append(line.bottom.xy)
    point_cloud = np.array(point_cloud)
    x_min = np.min(point_cloud[:,0])
    y_min = np.min(point_cloud[:,1])
    x_base = np.max(point_cloud[:,0]) - x_min
    y_base = np.max(point_cloud[:,1]) - y_min
    return minmax_norm(lines, x_min, x_base, y_min, y_base)

def norm_by_global_action(pose_lines: Sequence[PoseLine], global_action_lines: Sequence[GlobalActionLine], fallback=False) -> Sequence[Sequence[PoseLine]]:
    if(len(global_action_lines) == 0):
        if(fallback):
            return [pose_lines]
        else:
            return [[]] # TODO: evaluate if maybe "return [pose_lines]" is better
    normed_poses_seq = []
    for ga_line in global_action_lines:
        if(len(pose_lines) == 0):
            normed_poses_seq.append([])
            continue
        normed_poses = []
        for pose_line in pose_lines:
            new_top = np.array(pose_line.top.xy).flatten() - np.array(ga_line.center.xy).flatten()
            new_bottom = np.array(pose_line.bottom.xy).flatten() - np.array(ga_line.center.xy).flatten()
            normed_poses.append(PoseLine(Point(new_top), Point(new_bottom)))
        normed_poses_seq.append(normed_poses)
    return normed_poses_seq