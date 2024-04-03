#include "torch/torch.h"
#include "torch/script.h"
#include "Detector.h"
#include <chrono>
#include<Python.h>


struct LaserScan {
    std::string timestamp;
    std::string Lar;
    std::string d;
    std::string laser_num;
    std::string raw_laser_scan; //完整的laserscna数据
    std::vector<float> laser_scan_angle;
    std::vector<float> laser_scan_dis;
};

LaserScan parseLaserScan(std::string data)
{
    int index = 0;
    LaserScan laserscan;
    int start_pos = data.find("[");

    while (start_pos != std::string::npos)
    {
        int end_pos = data.find("]", start_pos);
        std::string data_inside_brackets = data.substr(start_pos + 1, end_pos - start_pos - 1);
        if (index == 0)
        {
            laserscan.timestamp = data.substr(start_pos + 1, end_pos - start_pos - 1);
        }
        else if (index == 4)
        {
            laserscan.raw_laser_scan = data.substr(start_pos + 1, end_pos - start_pos - 1);
        }
        start_pos = data.find("[", end_pos);
        index += 1;
    }    
    // 
    index = 0;
    std::istringstream iss(laserscan.raw_laser_scan);
    while (std::getline(iss, data, '|')) {
        if (index - 3 > 0 && ~index & 1)
            //laserscan.laser_scan_angle.push_back(stof(data));
            laserscan.laser_scan_angle.push_back(stof(data) * M_PI / 180.0);
        else if (index - 3 > 0 && index & 1)
            laserscan.laser_scan_dis.push_back(stof(data));
        index += 1;
    }
    return laserscan;
}
int main()
{
    //Py_Initialize();

    //PyRun_SimpleString("import sys");
    //PyRun_SimpleString("import numpy as np");


    //PyRun_SimpleString("sys.path.append(\"C:/Users/seer/AppData/Local/Programs/Python/Python38/Lib/site-packages\")");

    //PyRun_SimpleString("from matplotlib.animation import FuncAnimation");
    //PyRun_SimpleString("import matplotlib.pyplot as plt"); /*调用python文件*/

    //PyRun_SimpleString("fig, ax = plt.subplots()");
    //PyRun_SimpleString("line, = ax.plot([], [], 'b*')");
    //PyRun_SimpleString("point,  = ax.plot([], [], marker='o', linestyle='None', markersize=20, markerfacecolor='none', markeredgecolor='r')");
    //PyRun_SimpleString("point_people,  = ax.plot([], [], marker='o', linestyle='None', markersize=20, markerfacecolor='none', markeredgecolor='b') ");
    //PyRun_SimpleString("line.set_data(np.linspace(0, 10, 100), np.cos(np.linspace(0, 10, 100)))");
    //PyRun_SimpleString("plt.show()"); /*调用python文件*/

    //Py_Finalize();

    torch::DeviceType device_type;
    device_type = torch::kCUDA;
    torch::Device device(device_type);

    std::string data = "[240314 165837.922][Lar][d] [Laser: 1][28376814886526|-90|90|0.9|-90|7.623|-89.1|7.621|-88.2|7.623|-87.3|7.629|-86.4|7.628|-85.5|7.634|-84.6|7.613|-83.7|7.067|-82.8|7.077|-81.9|7.091|-81|7.107|-80.1|7.129|-79.2|7.152|-78.3|7.171|-77.4|7.194|-76.5|7.222|-75.6|7.245|-74.7|7.27|-73.8|7.308|-72.9|7.345|-72|7.378|-71.1|7.405|-70.2|7.445|-69.3|7.471|-68.4|7.526|-67.5|7.564|-66.6|7.61|-65.7|7.663|-64.8|7.716|-63.9|7.771|-63|7.837|-62.1|7.906|-61.2|7.97|-60.3|8.041|-59.4|8.107|-58.5|8.177|-57.6|8.267|-56.7|8.35|-55.8|8.435|-54.9|8.536|-54|8.619|-53.1|8.722|-52.2|8.818|-51.3|8.916|-50.4|0|-49.5|7.517|-48.6|0|-47.7|9.403|-46.8|0|-45.9|9.7|-45|9.813|-44.1|1.457|-43.2|1.432|-42.3|1.412|-41.4|1.391|-40.5|1.371|-39.6|1.352|-38.7|1.337|-37.8|1.315|-36.9|1.305|-36|1.285|-35.1|1.269|-34.2|1.258|-33.3|1.246|-32.4|1.225|-31.5|1.216|-30.6|1.204|-29.7|1.191|-28.8|1.182|-27.9|0|-27|0.893|-26.1|0.891|-25.2|0.882|-24.3|0.876|-23.4|0.87|-22.5|0.865|-21.6|0.852|-20.7|0.858|-19.8|0.845|-18.9|0.838|-18|0.837|-17.1|0.825|-16.2|0.824|-15.3|0.824|-14.4|0.822|-13.5|0.825|-12.6|0.825|-11.7|0.819|-10.8|0.818|-9.9|0.808|-9|0.807|-8.1|0.813|-7.2|0.811|-6.3|0.804|-5.4|0.803|-4.5|0.809|-3.6|0.8|-2.7|0.801|-1.8|0.801|-0.9|0.792|0|0.796|0.9|0.804|1.8|0.803|2.7|0.8|3.6|0.801|4.5|0.809|5.4|0.815|6.3|0.816|7.2|0.815|8.1|0.81|9|0.815|9.9|0.815|10.8|0.825|11.7|0.823|12.6|0.818|13.5|0.825|14.4|0.826|15.3|0.826|16.2|0.829|17.1|0.838|18|0.844|18.9|0.854|19.8|0.862|20.7|0.867|21.6|0.869|22.5|0.878|23.4|0.877|24.3|0.889|25.2|0.899|26.1|0.9|27|0|27.9|6.726|28.8|6.878|29.7|7.061|30.6|6.907|31.5|6.975|32.4|7.05|33.3|7.331|34.2|7.287|35.1|7.291|36|7.366|36.9|7.434|37.8|7.519|38.7|11.187|39.6|11.068|40.5|10.859|41.4|10.643|42.3|10.455|43.2|10.289|44.1|10.127|45|9.965|45.9|9.813|46.8|9.67|47.7|9.537|48.6|9.4|49.5|9.272|50.4|9.15|51.3|9.04|52.2|8.93|53.1|8.822|54|8.723|54.9|8.63|55.8|0|56.7|0|57.6|8.364|58.5|8.284|59.4|8.206|60.3|8.142|61.2|6.222|62.1|7.998|63|7.939|63.9|7.873|64.8|7.809|65.7|7.765|66.6|7.725|67.5|0|68.4|0|69.3|7.585|70.2|7.536|71.1|0|72|0|72.9|7.07|73.8|7.036|74.7|7.022|75.6|7|76.5|6.975|77.4|6.973|78.3|6.938|79.2|6.921|80.1|6.925|81|6.894|81.9|6.875|82.8|6.868|83.7|6.857|84.6|6.843|85.5|6.823|86.4|6.828|87.3|6.817|88.2|6.81|89.1|6.819|90|6.816]";
    LaserScan laserscan = parseLaserScan(data);

    torch::Tensor scans = torch::tensor(laserscan.laser_scan_dis, device);
    torch::Tensor scan_phi = torch::tensor(laserscan.laser_scan_angle, device);

    //c10::Device device = c10::kCPU;

   Detector* dr = new Detector();
   auto ct = dr->scans_to_cutout_torch(scans, scan_phi);
   ct = ct.unsqueeze(0);

   std::cout << ct << std::endl;

   ct = ct.to("cuda");

   //torch::Tensor& det_xys = torch::Tensor({});
   //torch::Tensor& det_cls = torch::Tensor({});
   //torch::Tensor& instance_mask = torch::Tensor({});
   torch::jit::script::Module module = torch::jit::load("C:\\Projects\\Python\\2D_lidar_person_detection\\gpu.pt", device);
   std::vector<torch::jit::IValue> x;
   x.push_back(ct);
   for (int i = 0; i < 100; ++i) {
       auto start = std::chrono::steady_clock::now();
       auto val = module.forward(x);
       torch::Tensor pred_cls = val.toTuple()->elements()[0].toTensor();
       torch::Tensor pred_reg = val.toTuple()->elements()[1].toTensor();


       pred_cls = torch::sigmoid(pred_cls.index({ 0 }));
       pred_reg = pred_reg.index({0});


       dr->nms_predicted_center(scans, scan_phi, pred_cls, pred_reg);
       auto end = std::chrono::steady_clock::now();
       std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
       std::cout << "耗时: " << duration.count() << " 秒" << std::endl;
   }


    return 0;


}