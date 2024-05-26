# 项目说明

该项目是金朔硕士学位论文第一个创新点的代码部分。该项目中包含的所有内容均围绕一个神经网络展开，该神经网络是一个双物体神经隐式场，输入两个单视角扫描残缺点云及一个
空间中的查询点，网络能够估计两物体各自的完整形状，并分别给出两物体在查询点处的udf值。配合相应的后处理算法，可从残缺点云中重建出准确的交互平分面。

# 项目结构

|-- configs                 // 训练和推理所需的配置文件\
|-- dataset                 // 加载数据所需的文件\
|-- models                  // 网络模型文件\
|-- postprocess             // 一些后处理代码，例如从训练好的网络重建IBS、计算指标等\
|-- preprocess              // 一些前处理代码，例如获取训练所需的单视角扫描点云、udf gt、ibs gt等\
|-- render                  // 调用blender进行渲染\
|-- utils                   // 一些工具类\
|-- README.md               // 项目说明\
|-- train_IBSNet.py         // 训练代码\
|-- train_grasping_field    // 训练对比方法GFNet的代码\
|-- train_IMNet             // 训练对比方法IMNet的代码\
|-- environment.yml         // 运行代码所需的环境信息\

# 如何运行项目

- 根据environment.yml中的信息配置conda虚拟环境，其中pointnet2-ops需要在github上找合适的开源实现（该库需要编译cuda代码，因此需要找到与本地cuda版本兼容的实现）
- 修改./configs/specs_train.json中的DataSource为实际的数据集地址
- 根据实际情况调整NumEpochs、BatchSize等参数
- 运行train.py

# 如何获取训练所需的数据

该网络是一个神经隐式场，输入是两物体的残缺点云和一个查询点，输出是该查询点处的两个准确udf值，所需的训练数据包括以下几个部分：
- 具有真实遮挡关系的双物体单视角扫描点云，可通过./preprocess/get_scan_pcd.py获取
- 查询点，及每个查询点到mesh表面的距离（即udf），可通过./preprocess/generate_udf_data.py获取

此外，为了评估本方法及其他方法估计的交互平分面是否准确，还需要Mesh形式的ibs gt，可通过./preprocess/get_ibs.py获取

