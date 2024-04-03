#include "Detector.h"
#include "utils.h"
#include <vector>
using namespace std;

//
//Detector::Detector()
//{
//
//}
void canonical_to_global(torch::Tensor scan_r, torch::Tensor scan_phi, torch::Tensor dx, torch::Tensor dy, torch::Tensor & dets_r, torch::Tensor &dets_phi) {
    torch::Tensor tmp_y = scan_r + dy;
    torch::Tensor tmp_phi = torch::arctan2(dx, tmp_y);
    dets_phi = tmp_phi + scan_phi;
    dets_r = tmp_y / torch::cos(tmp_phi);
}

void rphi_to_xy(torch::Tensor &r, torch::Tensor &phi, torch::Tensor& pred_xs, torch::Tensor& pred_ys) {
    pred_xs = r* -torch::sin(phi);
    pred_ys = r * torch::cos(phi);
}


torch::Tensor Detector::scans_to_cutout_torch(
    torch::Tensor scans,
    torch::Tensor scan_phi,
    int stride,
    bool centered,
    bool fixed,
    float window_width,
    float window_depth,
    int num_cutout_pts,
    float padding_val,
    bool area_mode
) {
    int num_scans, num_pts;
    torch::IntArrayRef size = scans.sizes();

    if (size.size() == 1)
    {
        num_scans = 1;
        num_pts = size[0];
    }
    torch::Tensor pi = torch::tensor(M_PI);

    // Compute half_alpha
    torch::Tensor dists = fixed ? scans : scans.index({ -1 }).expand_as(scans);
    torch::Tensor result = scans.index({ torch::indexing::Slice(torch::indexing::None, torch::indexing::None, stride) });
    torch::Tensor half_alpha = torch::atan(0.5 * window_width / torch::clamp(dists, 1e-2));
    torch::Tensor delta_alpha = 2.0 * half_alpha / (num_cutout_pts - 1);
    torch::Tensor ang_step = torch::arange(num_cutout_pts, scans.device()).view({ num_cutout_pts, 1, 1 }) * delta_alpha;

    torch::Tensor ang_ct = scan_phi.index({ torch::indexing::Slice(torch::indexing::None, torch::indexing::None, stride) }) - half_alpha + ang_step;
    ang_ct = (ang_ct + pi) % (2.0 * pi) - pi;
    torch::Tensor inds_ct = (ang_ct - scan_phi.index({ 0 })) / (scan_phi.index({ 1 }) - scan_phi.index({ 0 }));
    torch::Tensor outbound_mask = torch::logical_xor(inds_ct.lt(0), inds_ct.gt(num_pts - 1));

    torch::Tensor inds_ct_low = inds_ct.floor().toType(torch::kLong).clamp(0, num_pts - 1);
    torch::Tensor inds_ct_high = inds_ct.ceil().toType(torch::kLong).clamp(0, num_pts - 1);
    torch::Tensor inds_ct_ratio = (inds_ct - inds_ct_low).clamp(0, num_pts - 1);
    torch::Tensor ct_low = torch::gather(scans.expand_as(inds_ct_low), 2, inds_ct_low);
    torch::Tensor ct_high = torch::gather(scans.expand_as(inds_ct_high), 2, inds_ct_high);
    torch::Tensor ct = ct_low + inds_ct_ratio * (ct_high - ct_low);

    // use area sampling for down - sampling(close points)
    if (area_mode)
    {
        torch::Tensor num_pts_in_window = inds_ct.index({ -1 }) - inds_ct.index({ 0 });
        torch::Tensor area_mask = num_pts_in_window > num_cutout_pts;
        if (area_mask.sum().item().toInt() > 0)
        {
            int s_area = (num_pts_in_window.max() / num_cutout_pts).ceil().item().toInt();
            int num_ct_pts_area = s_area * num_cutout_pts;
            torch::Tensor delta_alpha_area = 2.0 * half_alpha / (num_ct_pts_area - 1);
            torch::Tensor ang_step_area = torch::arange(num_ct_pts_area, scans.device()).view({ num_ct_pts_area, 1, 1 }).mul(delta_alpha_area);//MUL?
            torch::Tensor ang_ct_area = scan_phi.slice(0, 0, scan_phi.size(0), stride) - half_alpha + ang_step_area;
            ang_ct_area = (ang_ct_area + M_PI).fmod(2.0 * M_PI).sub(M_PI);

            // 计算 inds_ct_area
            torch::Tensor inds_ct_area = ((ang_ct_area - scan_phi[0]) / (scan_phi[1] - scan_phi[0]));
            inds_ct_area = inds_ct_area
                .round()
                .toType(torch::kLong)
                .clamp(0, num_pts - 1);
            torch::Tensor ct_area = scans.expand_as(inds_ct_area).gather(2, inds_ct_area);

            ct_area = ct_area.view({ num_cutout_pts, s_area, num_scans, num_pts }).mean(1);

            // 更新 ct

            ct.index_put_({ "...",area_mask }, ct_area.index({ "...",area_mask }));
        }

    }
    ct.masked_fill_(outbound_mask, padding_val);

    // Apply clamp using torch.where
    ct = torch::where(ct < (dists - window_depth), dists - window_depth, ct);
    ct = torch::where(ct > (dists + window_depth), dists + window_depth, ct);

    if (centered) {
        ct = ct - dists;
        ct = ct / window_depth;
    }
    ct = ct.permute({ 2, 1, 0 }).to(torch::kFloat).contiguous();

    return ct;
}


void Detector::nms_predicted_center(
    torch::Tensor scan_grid,
    torch::Tensor phi_grid,
    torch::Tensor pred_cls,
    torch::Tensor pred_reg,
    bool pred_reg_prev,
    float min_dist

)
{
        torch::Tensor det_xys;
        torch::Tensor det_cls;
        torch::Tensor instance_mask;
    if (pred_cls.sizes()[0] != 1)
    {
        torch::Tensor pred_r, pred_phi;
        torch::Tensor pred_xs, pred_ys;
        canonical_to_global(scan_grid, phi_grid, pred_reg.index({ 0, "...",0 }), pred_reg.index({ 0, "...",1 }), pred_r, pred_phi);
        rphi_to_xy(pred_r, pred_phi, pred_xs, pred_ys);
        torch::Tensor sorted_indices = torch::argsort(pred_cls,0, /* descending = */ true);
        sorted_indices = sorted_indices.transpose(0, 1);
        //std::cout << sorted_indices << std::endl;
        //std::cout << pred_xs << std::endl;

        //std::cout << sorted_indices << std::endl;



        pred_xs = torch::index_select(pred_xs, 0, sorted_indices[{0}]);
        pred_ys = torch::index_select(pred_ys, 0, sorted_indices[{0}]);
        pred_cls = torch::index_select(pred_cls, 0, sorted_indices[{0}]);



        // compute pair - wise distance
        int num_pts = scan_grid.sizes()[0];
        torch::Tensor xdiff = pred_xs.view({ num_pts, 1 }) - pred_xs.view({1, num_pts});
        torch::Tensor ydiff = pred_ys.view({ num_pts, 1 }) - pred_ys.view({ 1, num_pts });
        torch::Tensor p_dist = torch::sqrt(torch::square(xdiff) + torch::square(ydiff));

        torch::Tensor keep = torch::ones(num_pts, torch::kBool);
        instance_mask = torch::zeros(num_pts, torch::kInt32);
        torch::Tensor instance_id = torch::tensor({ 1.0 });


        //std::cout << xdiff << std::endl;
        //std::cout << ydiff << std::endl;
        //std::cout << p_dist << std::endl;

        for (int i = 0; i < num_pts; i++)
        {
            if (!keep[i].item<bool>())
                continue;
            torch::Tensor dup_inds = p_dist.lt(min_dist);

            keep.index_put_({ dup_inds }, 0);//有问题
            keep.index_put_({ i }, 0);
            instance_mask.index_put({ sorted_indices.index({dup_inds}) }, instance_id);
            instance_id += 1;

            torch::Tensor stacked = torch::stack({ pred_xs, pred_ys }, 1);

            // 使用 keep 做逻辑索引，获取满足条件的元素
            det_xys = stacked.index({ keep.nonzero().squeeze(1) });
            det_cls = torch::index_select(pred_cls, /* dim = */ 0, keep);
        }
    }
}