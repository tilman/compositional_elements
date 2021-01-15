from compoelem.types import *
from compoelem.config import config

#optimize with KD trees or faiss: https://github.com/facebookresearch/faiss

# def euclidean_distance(a: Keypoint, b: Keypoint) -> float:
#     return 0

# def compare_global_action_line(a: GlobalActionLine], b: GlobalActionLine) -> float:
#     return 0

# def compare_global_action_lines(a: Sequence[GlobalActionLine], b: Sequence[GlobalActionLine]) -> float:
#     return 0

def compare_pose_line(a: PoseLine, b: PoseLine) -> float:
    return np.mean([a.top.distance(b.top), a.bottom.distance(b.bottom)])

def find_nearest_pose_line_distance(target: PoseLine, lines: Sequence[PoseLine]) -> Tuple[float, int]:
    # if(len(lines)):
    #     return config["compare"]["no_pose_line_penalty"]
    comparisons: Sequence[Tuple[float, int]] = []
    for idx, line in enumerate(lines):
        mean_distance = compare_pose_line(line, target)
        comparisons.append((mean_distance, idx))
    comparisons = np.array(comparisons)
    nearest_pose_line = comparisons[np.argmin(comparisons[:,0])]
    return nearest_pose_line

def compare_pose_lines(a: Sequence[PoseLine], b: Sequence[PoseLine]) -> float:
    if(len(b) == 0):
        return len(a) * config["compare"]["no_pose_line_penalty"] 
    nearest_pose_line_distances: Sequence[Tuple[float, int]] = []
    for query_pose_line in a:
        nearest_pose_line_distances.append(find_nearest_pose_line_distance(query_pose_line, b))
    nearest_pose_line_distances = np.array(nearest_pose_line_distances)
    nearest_pose_line_sum = np.sum(nearest_pose_line_distances[:, 0])
    penalty_pose_line_indices = [idx for idx in range(0,len(b)) if idx not in nearest_pose_line_distances[:,1]]
    penalty_pose_lines: Sequence[PoseLine] = np.array(b)[penalty_pose_line_indices]
    if len(penalty_pose_lines) > 0:
        penalty_sum = np.sum(np.array([find_nearest_pose_line_distance(line, a) for line in penalty_pose_lines])[:,0]) 
        return nearest_pose_line_sum + penalty_sum
    else:
        return nearest_pose_line_sum