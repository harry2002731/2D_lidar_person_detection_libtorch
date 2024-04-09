#pragma once
#include "torch/torch.h"
#include "torch/script.h"
class Detector
{
public:
    torch::Tensor scans_to_cutout_torch(
        torch::Tensor scans,
        torch::Tensor scan_phi,
        int stride = 1,
        bool centered = true,
        bool fixed = true,
        float window_width = 1.0,
        float window_depth = 0.5,
        int num_cutout_pts = 56,
        float padding_val = 29.99,
        bool area_mode = true
    );
    void nms_predicted_center(
        torch::Tensor scan_grid,
        torch::Tensor phi_grid,
        torch::Tensor pred_cls,
        torch::Tensor pred_reg,
        torch::Tensor& dets_xys,
        torch::Tensor& dets_cls,
        bool pred_reg_prev = 0,
        float min_dist = 0.5


    );
};

