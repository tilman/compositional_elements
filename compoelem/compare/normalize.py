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

def minmax_norm_by_imgrect(lines: Sequence[PoseLine], file_path: str) -> Sequence[PoseLine]:
    img = cv2.imread(file_path)
    img = converter.resize(img)
    y_base, x_base, _ = img.shape #type: ignore
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