# 2D_lidar_person_detection_libtorch

2D_lidar_person_detection_libtorch 是基于2D_lidar_person_detection迁移到libtorch的版本。

## Installation

libtorch和pytorch两者的版本需要一致。

使用[pip](https://pip.pypa.io/en/stable/)安装pytorch.

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

libtorch安装

1. 下载 https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-2.2.1%2Bcu118.zip
2. 将libtorch在visual studio中进行部署 [LibTorch的安装、配置与使用-CSDN博客](https://blog.csdn.net/weixin_45632168/article/details/114679263)

    问题汇总
    1. [C++部署Pytorch（Libtorch）出现问题、错误汇总](https://blog.csdn.net/zzz_zzz12138/article/details/109138805)

    2. [cuda链接异常](https://blog.csdn.net/qq_36038453/article/details/120278523)

## Test installation

libtorch测试,若环境配置正确则会输出正确结果
```C++
#include "torch/torch.h"
#include "torch/script.h"

int main()
{
    torch::Tensor output = torch::zeros({ 3,2 });
    std::cout << output << std::endl;
    return 0;
}

```

## Usage

1. 在Python中导出torchscript模型，需要在原2D_lidar_person_detection中添加以下代码。可以参考我基于2D_lidar_person_detection[修改的代码](https://github.com/harry2002731/2D_lidar_person_detection)。

```python
    #将模型TorchScript化
    def convert(self):
        traced_script_module = torch.jit.trace(self._model, self.ct_) # _model 为 nn.Module类型，ct_为传入的数据
        output = traced_script_module(self.ct_)
        traced_script_module.save('cpu.pt')
```

2. 在C++中修改路径
```C++

    //模型推理
    torch::jit::script::Module module = torch::jit::load(".\\cpu.pt", device);
    auto val = module.forward(x);
    torch::Tensor pred_cls = val.toTuple()->elements()[0].toTensor();
    torch::Tensor pred_reg = val.toTuple()->elements()[1].toTensor();
    pred_cls = torch::sigmoid(pred_cls.index({ 0 }));
    pred_reg = pred_reg.index({ 0 });
```

3. 运行测试输出！

## License

[MIT](https://choosealicense.com/licenses/mit/)