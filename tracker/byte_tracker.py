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

class ByteTrackerMotion(BaseTracker):
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
                 num_tentatives: int = 3, ground_assign: bool = False, use_motion: bool = False, use_kalman_filter: bool = False,
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

        # print("Call ByteTracker.track()")

        frame_id = data_sample["frame_id"]
        bboxes = data_sample["bboxes"]
        labels = data_sample["labels"]
        scores = data_sample["scores"]
        homography = data_sample["homography"]
        motion_vec = data_sample["motion"]

        
        #         if True:
        # point_ground = get_ground_point_from_bbox(bboxes, homography)

        # print(data_sample.pred_instances.__dict__)
        # print('bboxes', bboxes.shape)
        # print('labels', labels.shape)
        # print('scores', scores.shape)

        # frame_id = metainfo.get('frame_id', -1)

        # print('frame_id', frame_id)
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
            # 0. init
            ids = torch.full((bboxes.size(0), ),
                             -1,
                             dtype=labels.dtype,
                             device=labels.device)

            # get the detection bboxes for the first association
            first_det_inds = scores > self.obj_score_thrs['high']
            first_det_bboxes = bboxes[first_det_inds]
            first_det_labels = labels[first_det_inds]
            first_det_scores = scores[first_det_inds]
            first_det_ids = ids[first_det_inds]
            first_det_motion = motion_vec[first_det_inds] if self.use_motion else None

            # get the detection bboxes for the second association
            second_det_inds = (~first_det_inds) & (
                scores > self.obj_score_thrs['low'])
            second_det_bboxes = bboxes[second_det_inds]
            second_det_labels = labels[second_det_inds]
            second_det_scores = scores[second_det_inds]
            second_det_ids = ids[second_det_inds]
            second_det_motion = motion_vec[second_det_inds] if self.use_motion else None
            

            # 1. use Kalman Filter to predict current location
            for id in self.confirmed_ids:
                # track is lost in previous frame
                if self.tracks[id].frame_ids[-1] != frame_id - 1:
                    self.tracks[id].mean[7] = 0
                (self.tracks[id].mean,
                 self.tracks[id].covariance) = self.kf.predict(
                     self.tracks[id].mean, self.tracks[id].covariance)

            # 2. first match
            first_match_track_inds, first_match_det_inds = self.assign_ids(
                self.confirmed_ids, first_det_bboxes, first_det_labels,
                first_det_scores, homography, first_det_motion, self.weight_iou_with_det_scores,
                self.match_iou_thrs['high'])
            # '-1' mean a detection box is not matched with tracklets in
            # previous frame
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

            # 3. use unmatched detection bboxes from the first match to match
            # the unconfirmed tracks
            (tentative_match_track_inds,
             tentative_match_det_inds) = self.assign_ids(
                 self.unconfirmed_ids, first_unmatch_det_bboxes,
                 first_unmatch_det_labels, first_unmatch_det_scores, homography, first_unmatch_det_motion,
                 self.weight_iou_with_det_scores,
                 self.match_iou_thrs['tentative'])
            valid = tentative_match_det_inds > -1
            first_unmatch_det_ids[valid] = torch.tensor(self.unconfirmed_ids)[
                tentative_match_det_inds[valid]].to(labels)

            # 4. second match for unmatched tracks from the first match
            first_unmatch_track_ids = []
            for i, id in enumerate(self.confirmed_ids):
                # tracklet is not matched in the first match
                case_1 = first_match_track_inds[i] == -1
                # tracklet is not lost in the previous frame
                case_2 = self.tracks[id].frame_ids[-1] == frame_id - 1
                if case_1 and case_2:
                    first_unmatch_track_ids.append(id)

            second_match_track_inds, second_match_det_inds = self.assign_ids(
                first_unmatch_track_ids, second_det_bboxes, second_det_labels,
                second_det_scores, homography, second_det_motion, False, self.match_iou_thrs['low'])
            valid = second_match_det_inds > -1
            second_det_ids[valid] = torch.tensor(first_unmatch_track_ids)[
                second_match_det_inds[valid]].to(ids)

            # 5. gather all matched detection bboxes from step 2-4
            # we only keep matched detection bboxes in second match, which
            # means the id != -1
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

            # 6. assign new ids
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

        # print("After update")
        # print('bboxes', bboxes.shape)
        # print('labels', labels.shape)
        # print('scores', scores.shape)
        # print('ids', ids.shape)

        # update pred_track_instances
        pred_track_instances = {}
        pred_track_instances["bboxes"] = bboxes
        pred_track_instances["labels"] = labels
        pred_track_instances["scores"] = scores
        pred_track_instances["instances_id"] = ids

        return pred_track_instances