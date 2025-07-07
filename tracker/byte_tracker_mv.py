# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

try:
    import lap
except ImportError:
    lap = None
    
import numpy as np
import torch

from tracker.base_tracker import BaseTracker
from tracker.kalman_filter import KalmanFilter
from tracker.utils import bbox_cxcyah_to_xyxy, bbox_overlaps, bbox_overlaps_ground, bbox_xyxy_to_cxcyah

class ByteTrackerMV(BaseTracker):
    """Tracker for ByteTrack.

    Args:
        motion (dict): Configuration of motion. Defaults to None.
        obj_score_thrs (dict): Detection score threshold for matching objects.
            - high (float): Threshold of the first matching. Defaults to 0.6.
            - low (float): Threshold of the second matching. Defaults to 0.1.
        init_track_thr (float): Detection score threshold for initializing a
            new tracklet. Defaults to 0.7.
        weight_iou_with_det_scores (bool): Whether using detection scores to
            weight IOU which is used for matching. Defaults to True.
        match_iou_thrs (dict): IOU distance threshold for matching between two
            frames.
            - high (float): Threshold of the first matching. Defaults to 0.1.
            - low (float): Threshold of the second matching. Defaults to 0.5.
            - tentative (float): Threshold of the matching for tentative
                tracklets. Defaults to 0.3.
        num_tentatives (int, optional): Number of continuous frames to confirm
            a track. Defaults to 3.
    """

    
    def __init__(self,
                 motion: Optional[dict] = None,
                 obj_score_thrs: dict = dict(high=0.6, low=0.1),
                 init_track_thr: float = 0.7,
                 weight_iou_with_det_scores: bool = True,
                 match_iou_thrs: dict = dict(high=0.1, low=0.5, tentative=0.3),
                 num_tentatives: int = 3, ground_assign: bool = False, use_motion: bool = False, use_kalman_filter: bool = True,
                 **kwargs):
        super().__init__(**kwargs)

        if lap is None:
            raise RuntimeError('lap is not installed,\
                 please install it by: pip install lap')
        
        self.motion = KalmanFilter()

        self.obj_score_thrs = obj_score_thrs
        self.init_track_thr = init_track_thr

        self.weight_iou_with_det_scores = weight_iou_with_det_scores
        self.match_iou_thrs = match_iou_thrs

        self.num_tentatives = num_tentatives
        self.ground_assign = ground_assign
        self.use_motion = use_motion
        self.use_kalman_filter = use_kalman_filter

    @property
    def confirmed_ids(self) -> List:
        """Confirmed ids in the tracker."""
        ids = [id for id, track in self.tracks.items() if not track.tentative]
        return ids

    @property
    def unconfirmed_ids(self) -> List:
        """Unconfirmed ids in the tracker."""
        ids = [id for id, track in self.tracks.items() if track.tentative]
        return ids

    def init_track(self, id: int, obj: Tuple[torch.Tensor]) -> None:
        """Initialize a track."""
        super().init_track(id, obj)
        if self.tracks[id].frame_ids[-1] == 0:
            self.tracks[id].tentative = False
        else:
            self.tracks[id].tentative = True
        bbox = bbox_xyxy_to_cxcyah(self.tracks[id].bboxes[-1])  # size = (1, 4)
        assert bbox.ndim == 2 and bbox.shape[0] == 1
        bbox = bbox.squeeze(0).cpu().numpy()
        self.tracks[id].mean, self.tracks[id].covariance = self.kf.initiate(
            bbox)

    def update_track(self, id: int, obj: Tuple[torch.Tensor]) -> None:
        """Update a track."""
        super().update_track(id, obj)
        if self.tracks[id].tentative:
            if len(self.tracks[id]['bboxes']) >= self.num_tentatives:
                self.tracks[id].tentative = False
        bbox = bbox_xyxy_to_cxcyah(self.tracks[id].bboxes[-1])  # size = (1, 4)
        assert bbox.ndim == 2 and bbox.shape[0] == 1
        bbox = bbox.squeeze(0).cpu().numpy()
        track_label = self.tracks[id]['labels'][-1]
        label_idx = self.memo_items.index('labels')
        obj_label = obj[label_idx]
        assert obj_label == track_label
        self.tracks[id].mean, self.tracks[id].covariance = self.kf.update(
            self.tracks[id].mean, self.tracks[id].covariance, bbox)

    def pop_invalid_tracks(self, frame_id: int) -> None:
        """Pop out invalid tracks."""
        invalid_ids = []
        for k, v in self.tracks.items():
            # case1: disappeared frames >= self.num_frames_retrain
            case1 = frame_id - v['frame_ids'][-1] >= self.num_frames_retain
            # case2: tentative tracks but not matched in this frame
            case2 = v.tentative and v['frame_ids'][-1] != frame_id
            if case1 or case2:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)
    
    def update_ious_with_motion(self, prev_track_bboxes, det_bboxes, det_motions, det_homography):
#         new_ious = torch.zeros_like(ious)
        
        if self.use_motion:
            det_bboxes_copy = torch.clone(det_bboxes)
            for i in range(len(prev_track_bboxes)):
                for j in range(len(det_bboxes)):
    #                 cur_iou = ious[i, j]
                    kalman_bbox = prev_track_bboxes[i]
                    prev_bbox =  prev_track_bboxes[i]

                    det_bbox = det_bboxes[j]
                    det_motion = det_motions[j]

                    if det_motion[0] is not None and ~torch.isnan(det_motion[0]):
                        
                        det_bboxes_copy[j] = det_bboxes[j] + det_motion.repeat(2)
#                         print(det_bboxes[j], det_bboxes_copy[j], det_motion)
    #                 print("kalman", kalman_bbox)
    #                 print("det_bbox", det_bbox)
    #                 print("prev_bbox", prev_bbox)
    #                 print("det_motion", det_motion)
    #                 print("det_bbox_shifted", det_bboxes_copy[j])


    #                 new_ious[i,j] = cur_iou
        else:
            det_bboxes_copy = det_bboxes
        
        if self.ground_assign:
            new_ious = bbox_overlaps_ground(prev_track_bboxes, det_bboxes_copy, det_homography)
        else:
            new_ious = bbox_overlaps(prev_track_bboxes, det_bboxes_copy)
                    
        
        return new_ious
    
    def assign_ids(
            self,
            ids: List[int],
            det_bboxes: torch.Tensor,
            det_labels: torch.Tensor,
            det_scores: torch.Tensor,
            det_homography,
            det_motion,
            weight_iou_with_det_scores: Optional[bool] = False,
            match_iou_thr: Optional[float] = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assign ids.

        Args:
            ids (list[int]): Tracking ids.
            det_bboxes (Tensor): of shape (N, 4)
            det_labels (Tensor): of shape (N,)
            det_scores (Tensor): of shape (N,)
            weight_iou_with_det_scores (bool, optional): Whether using
                detection scores to weight IOU which is used for matching.
                Defaults to False.
            match_iou_thr (float, optional): Matching threshold.
                Defaults to 0.5.

        Returns:
            tuple(np.ndarray, np.ndarray): The assigning ids.
        """
        if  self.use_kalman_filter:
            # get track_bboxes from kalman
            track_bboxes = np.zeros((0, 4))
            for id in ids:
                track_bboxes = np.concatenate(
                    (track_bboxes, self.tracks[id].mean[:4][None]), axis=0)
            track_bboxes = torch.from_numpy(track_bboxes).to(det_bboxes)
            track_bboxes = bbox_cxcyah_to_xyxy(track_bboxes)
        else:
            #get track_bboxes as previous boxes
            track_bboxes = np.zeros((0, 4))
            for id in ids:
                track_bboxes = np.concatenate(
                    (track_bboxes, self.tracks[id].bboxes[-1]), axis=0)
            track_bboxes = torch.from_numpy(track_bboxes).to(det_bboxes)
        
        # #get prev track_bboxes
        # prev_track_bboxes = np.zeros((0, 4))
        # for id in ids:
        #     prev_track_bboxes = np.concatenate(
        #         (prev_track_bboxes, self.tracks[id].bboxes[-1]), axis=0)
        # prev_track_bboxes = torch.from_numpy(prev_track_bboxes).to(det_bboxes)

        # compute distance
#         if self.ground_assign:
#             ious = bbox_overlaps_ground(track_bboxes, det_bboxes, det_homography)
#         else:
#             ious = bbox_overlaps(track_bboxes, det_bboxes)
        
        ious = self.update_ious_with_motion(track_bboxes, det_bboxes, det_motion, det_homography)
            

            
        if weight_iou_with_det_scores:
            ious *= det_scores
        # support multi-class association
        track_labels = torch.tensor([
            self.tracks[id]['labels'][-1] for id in ids
        ]).to(det_bboxes.device)

        cate_match = det_labels[None, :] == track_labels[:, None]
        # to avoid det and track of different categories are matched
        cate_cost = (1 - cate_match.int()) * 1e6

        dists = (1 - ious + cate_cost).cpu().numpy()

        # bipartite match
        if dists.size > 0:
            cost, row, col = lap.lapjv(
                dists, extend_cost=True, cost_limit=1 - match_iou_thr)
        else:
            row = np.zeros(len(ids)).astype(np.int32) - 1
            col = np.zeros(len(det_bboxes)).astype(np.int32) - 1
        return row, col

    def merge_multiview_bboxes(self, bboxes, world_points, view_ids, labels, scores, dist_thresh=0.8, bbox_radius=1.5):
        """
        Merge multiview bboxes based on world point coordinates if the distance
        between any pair of points from different views is less than dist_thresh.
        From the resulting merge, create a "ground" bounding box centered at the
        mean of the merged world points. The label and score of this new bounding
        box come from the merged detections (label of the first detection in the
        cluster, and maximum score among them). It's not possible to merge
        world points from the same view. Finally, return the merged bboxes,
        labels, scores, and a mapping from each merged (ground) bbox to its
        original bboxes.

        Args:
            bboxes (Tensor): Shape (N, 4).
            world_points (Tensor): Shape (N, 2).
            view_ids (Tensor): Shape (N, ).
            labels (Tensor): Shape (N, ).
            scores (Tensor): Shape (N, ).
            dist_thresh (float): Threshold for merging points.

        Returns:
            merged_bboxes (Tensor): Merged bounding boxes of shape (M, 4).
            merged_labels (Tensor): Merged labels of shape (M, ).
            merged_scores (Tensor): Merged scores of shape (M, ).
            mapping_groundbbox_to_bboxes (dict): Dictionary keyed by merged bbox
                index. Each value is a dict with:
                "inds": the indices of the original bboxes that formed this cluster,
                "orig_bboxes": the original bboxes for these detections.
        """
        device = bboxes.device
        bboxes_np = bboxes.detach().cpu().numpy()
        points_np = world_points.detach().cpu().numpy()
        views_np = view_ids.detach().cpu().numpy()
        labels_cpu = labels.detach().cpu()
        scores_cpu = scores.detach().cpu()

        N = points_np.shape[0]
        visited = [False] * N
        adjacency = [[] for _ in range(N)]

        # Build adjacency list based on distance < dist_thresh and different views
        import numpy as np
        for i in range(N):
            for j in range(i + 1, N):
                if views_np[i] != views_np[j]:
                    dist_ij = np.linalg.norm(points_np[i] - points_np[j])
                    if dist_ij < dist_thresh:
                        adjacency[i].append(j)
                        adjacency[j].append(i)

        # Find clusters using BFS
        clusters = []
        for i in range(N):
            if not visited[i]:
                queue = [i]
                visited[i] = True
                cluster = [i]

                while queue:
                    cur = queue.pop()
                    for nb in adjacency[cur]:
                        if not visited[nb]:
                            visited[nb] = True
                            cluster.append(nb)
                            queue.append(nb)
                clusters.append(cluster)

        merged_bboxes = []
        merged_labels = []
        merged_scores = []
        mapping_groundbbox_to_bboxes = {}

        # For each cluster, create one "ground" bounding box
        for c_idx, cluster_inds in enumerate(clusters):
            cluster_points = points_np[cluster_inds]  # shape [k, 2]
            # print(f"cluster_points: {cluster_points.shape}")
            mean_point = cluster_points.mean(axis=0)
            cx, cy, _ = mean_point
            # Create a small bounding box around the mean point
            # (these coordinates are arbitrary for illustration)
            x1, y1 = cx - bbox_radius, cy - bbox_radius
            x2, y2 = cx + bbox_radius, cy + bbox_radius

            # Label is taken from the first detection in the cluster
            # Score is the maximum among them
            cluster_labels = labels_cpu[cluster_inds]
            cluster_scores = scores_cpu[cluster_inds]
            label_val = cluster_labels[0]
            score_val = cluster_scores.max()

            merged_bboxes.append([x1, y1, x2, y2])
            merged_labels.append(label_val)
            merged_scores.append(score_val)

            # Store original bboxes for later mapping
            cluster_original_bboxes = bboxes[cluster_inds]  # shape [k, 4]
            mapping_groundbbox_to_bboxes[c_idx] = {
                "inds": cluster_inds,
                "orig_bboxes": cluster_original_bboxes
            }

        merged_bboxes = torch.tensor(merged_bboxes,
                                     dtype=bboxes.dtype,
                                     device=device)
        merged_labels = torch.stack(merged_labels).to(device)
        merged_scores = torch.stack(merged_scores).to(device)

        return merged_bboxes, merged_labels, merged_scores, mapping_groundbbox_to_bboxes

    def map_ground_bbox_to_bboxes(self, bboxes, labels, scores, ids, mapping_groundbbox_to_bboxes):
        """
        Map the merged "ground" bboxes back to the original bboxes. For each merged
        bbox, replicate its label, score, and assigned ID to all original bounding
        boxes that formed that merge. Return the expanded set of bboxes, labels,
        scores, and ids.

        Args:
            bboxes (Tensor): Merged bounding boxes of shape (M, 4).
            labels (Tensor): Merged labels of shape (M, ).
            scores (Tensor): Merged scores of shape (M, ).
            ids (Tensor): Merged track IDs of shape (M, ).
            mapping_groundbbox_to_bboxes (dict): Dictionary from merge_multiview_bboxes.

        Returns:
            out_bboxes (Tensor): Original bounding boxes of shape (N, 4).
            out_labels (Tensor): Expanded labels of shape (N, ).
            out_scores (Tensor): Expanded scores of shape (N, ).
            out_ids (Tensor): Expanded ids of shape (N, ).
        """
        device = bboxes.device
        out_bboxes = []
        out_labels = []
        out_scores = []
        out_ids = []

        # Each key in the dictionary corresponds to one merged (ground) bbox index.
        # Replicate that ground bbox's label, score, and id to each original bbox.
        for c_idx, cluster_info in mapping_groundbbox_to_bboxes.items():
            cluster_original_bboxes = cluster_info["orig_bboxes"]  # shape [k, 4]
            label_val = labels[c_idx]    # shape []
            score_val = scores[c_idx]
            id_val = ids[c_idx]

            # For each original bbox in the cluster, assign the same label, score, id
            for i in range(cluster_original_bboxes.size(0)):
                out_bboxes.append(cluster_original_bboxes[i])
                out_labels.append(label_val)
                out_scores.append(score_val)
                out_ids.append(id_val)

        out_bboxes = torch.stack(out_bboxes, dim=0).to(device)
        out_labels = torch.stack(out_labels, dim=0).to(device)
        out_scores = torch.stack(out_scores, dim=0).to(device)
        out_ids = torch.stack(out_ids, dim=0).to(device)

        return out_bboxes, out_labels, out_scores, out_ids

    def track(self, data_sample, **kwargs):
        """Tracking forward function.

        Args:
            data_sample (:obj:`DetDataSample`): The data sample.
                It includes information such as `pred_instances`.

        Returns:
            :obj:`InstanceData`: Tracking results of the input images.
            Each InstanceData usually contains ``bboxes``, ``labels``,
            ``scores`` and ``instances_id``.
        """

        frame_id = data_sample["frame_id"]
        bboxes = data_sample["bboxes"]
        world_points = data_sample["world_points"]
        view_ids = data_sample["view_ids"]
        labels = data_sample["labels"]
        scores = data_sample["scores"]
        homography = data_sample["homography"]
        motion_vec = data_sample["motion"]

        # Merge multiview world points to create bboxes
        bboxes, labels, scores, mapping_groundbbox_to_bboxes = self.merge_multiview_bboxes(
            bboxes, world_points, view_ids, labels, scores, dist_thresh=0.8, bbox_radius=2.5)

        if frame_id == 0:
            self.reset()
        if not hasattr(self, 'kf'):
            self.kf = self.motion

        if self.empty or bboxes.size(0) == 0:
            valid_inds = scores > self.init_track_thr
            scores = scores[valid_inds]
            bboxes = bboxes[valid_inds]
            labels = labels[valid_inds]
            num_new_tracks = bboxes.size(0)
            ids = torch.arange(self.num_tracks,
                               self.num_tracks + num_new_tracks).to(labels)
            self.num_tracks += num_new_tracks

        else:
            ids = torch.full((bboxes.size(0), ),
                             -1,
                             dtype=labels.dtype,
                             device=labels.device)

            first_det_inds = scores > self.obj_score_thrs['high']
            first_det_bboxes = bboxes[first_det_inds]
            first_det_labels = labels[first_det_inds]
            first_det_scores = scores[first_det_inds]
            first_det_ids = ids[first_det_inds]
            first_det_motion = motion_vec[first_det_inds] if self.use_motion else None

            second_det_inds = (~first_det_inds) & (
                scores > self.obj_score_thrs['low'])
            second_det_bboxes = bboxes[second_det_inds]
            second_det_labels = labels[second_det_inds]
            second_det_scores = scores[second_det_inds]
            second_det_ids = ids[second_det_inds]
            second_det_motion = motion_vec[second_det_inds] if self.use_motion else None

            for id in self.confirmed_ids:
                if self.tracks[id].frame_ids[-1] != frame_id - 1:
                    self.tracks[id].mean[7] = 0
                (self.tracks[id].mean,
                 self.tracks[id].covariance) = self.kf.predict(
                     self.tracks[id].mean, self.tracks[id].covariance)

            first_match_track_inds, first_match_det_inds = self.assign_ids(
                self.confirmed_ids, first_det_bboxes, first_det_labels,
                first_det_scores, homography, first_det_motion,
                self.weight_iou_with_det_scores,
                self.match_iou_thrs['high'])
            valid = first_match_det_inds > -1
            first_det_ids[valid] = torch.tensor(
                self.confirmed_ids)[first_match_det_inds[valid]].to(labels)

            first_match_det_bboxes = first_det_bboxes[valid]
            first_match_det_labels = first_det_labels[valid]
            first_match_det_scores = first_det_scores[valid]
            first_match_det_ids = first_det_ids[valid]
            assert (first_match_det_ids > -1).all()

            first_unmatch_det_bboxes = first_det_bboxes[~valid]
            first_unmatch_det_labels = first_det_labels[~valid]
            first_unmatch_det_scores = first_det_scores[~valid]
            first_unmatch_det_ids = first_det_ids[~valid]
            first_unmatch_det_motion = first_det_motion[~valid] if self.use_motion else None
            assert (first_unmatch_det_ids == -1).all()

            (tentative_match_track_inds,
             tentative_match_det_inds) = self.assign_ids(
                 self.unconfirmed_ids, first_unmatch_det_bboxes,
                 first_unmatch_det_labels, first_unmatch_det_scores,
                 homography, first_unmatch_det_motion,
                 self.weight_iou_with_det_scores,
                 self.match_iou_thrs['tentative'])
            valid = tentative_match_det_inds > -1
            first_unmatch_det_ids[valid] = torch.tensor(self.unconfirmed_ids)[
                tentative_match_det_inds[valid]].to(labels)

            first_unmatch_track_ids = []
            for i, id in enumerate(self.confirmed_ids):
                case_1 = first_match_track_inds[i] == -1
                case_2 = self.tracks[id].frame_ids[-1] == frame_id - 1
                if case_1 and case_2:
                    first_unmatch_track_ids.append(id)

            second_match_track_inds, second_match_det_inds = self.assign_ids(
                first_unmatch_track_ids, second_det_bboxes, second_det_labels,
                second_det_scores, homography, second_det_motion, False,
                self.match_iou_thrs['low'])
            valid = second_match_det_inds > -1
            second_det_ids[valid] = torch.tensor(first_unmatch_track_ids)[
                second_match_det_inds[valid]].to(ids)

            valid = second_det_ids > -1
            bboxes = torch.cat(
                (first_match_det_bboxes, first_unmatch_det_bboxes), dim=0)
            bboxes = torch.cat((bboxes, second_det_bboxes[valid]), dim=0)

            labels = torch.cat(
                (first_match_det_labels, first_unmatch_det_labels), dim=0)
            labels = torch.cat((labels, second_det_labels[valid]), dim=0)

            scores = torch.cat(
                (first_match_det_scores, first_unmatch_det_scores), dim=0)
            scores = torch.cat((scores, second_det_scores[valid]), dim=0)

            ids = torch.cat((first_match_det_ids, first_unmatch_det_ids),
                            dim=0)
            ids = torch.cat((ids, second_det_ids[valid]), dim=0)

            new_track_inds = ids == -1
            ids[new_track_inds] = torch.arange(
                self.num_tracks,
                self.num_tracks + new_track_inds.sum()).to(labels)
            self.num_tracks += new_track_inds.sum()

        self.update(
            ids=ids,
            bboxes=bboxes,
            scores=scores,
            labels=labels,
            frame_ids=frame_id)

        pred_track_instances = {}
        # Map the ground bboxes back to original bboxes, duplicating track info
        bboxes, labels, scores, ids = self.map_ground_bbox_to_bboxes(
            bboxes, labels, scores, ids, mapping_groundbbox_to_bboxes)
        pred_track_instances["bboxes"] = bboxes
        pred_track_instances["labels"] = labels
        pred_track_instances["scores"] = scores
        pred_track_instances["instances_id"] = ids

        return pred_track_instances