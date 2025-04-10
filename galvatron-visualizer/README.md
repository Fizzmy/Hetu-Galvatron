# Galvatron Memory Visualizer | Galvatron内存可视化工具

[English](#english) | [中文](#chinese)

<a name="english"></a>
## Galvatron Memory Visualizer

### Introduction

Galvatron Memory Visualizer is an interactive tool for analyzing and visualizing memory usage in large language models. Based on the Galvatron memory cost model, this tool provides users with intuitive visual representations of memory allocation for different model configurations and distributed training strategies.

### Key Features

- **Interactive Memory Visualization**: View memory allocation with interactive treemap visualization
- **Memory Distribution Analysis**: Analyze memory usage by category with bar charts and proportion views
- **Distributed Training Strategies**: Configure tensor parallelism, pipeline parallelism, and other distribution strategies
- **Real-time Memory Estimation**: Get instant memory usage feedback when changing parameters
- **Bilingual Support**: Full Chinese and English interface support
- **Configuration Upload**: Import Galvatron configuration files for precise memory analysis

### Memory Categories

The visualizer analyzes and displays memory usage across several categories:

- **Activation Memory**: Memory used for storing activations during the forward pass
- **Model States**: Combined memory for parameters, gradients, and optimizer states
  - **Parameter Memory**: Memory used to store model parameters
  - **Gradient Memory**: Memory used for gradients during backpropagation
  - **Optimizer Memory**: Memory used by optimizer states
  - **Gradient Accumulation**: Memory used for gradient accumulation in multi-step updates

### Installation

1. Clone the repository
```bash
git clone https://github.com/PKU-TANGENT/Hetu-Galvatron.git
cd Hetu-Galvatron/galvatron-visualizer
```

2. Install dependencies
```bash
npm install
```

3. Start the development server
```bash
npm start
```

4. Open [http://localhost:3000](http://localhost:3000) to view the application

### Usage

1. **Select a Configuration**: Choose a predefined model or upload a configuration file
2. **Adjust Parameters**: Modify model parameters in the config panel
3. **View Memory Analysis**: Observe memory allocation in the treemap visualization
4. **Analyze Distributions**: Use the bar chart and proportion views to understand memory usage patterns
5. **Explore Different Strategies**: Test different parallel strategies to optimize memory usage

### License

This project is part of the Hetu-Galvatron framework from Peking University. For more information, visit the [GitHub repository](https://github.com/PKU-DAIR/Hetu-Galvatron) or read the related [research paper](https://arxiv.org/abs/2211.13878).

<a name="chinese"></a>
## Galvatron内存可视化工具

### 项目介绍

Galvatron内存可视化工具是一个用于分析和可视化大型语言模型内存使用情况的交互式应用。基于Galvatron内存成本模型，该工具为用户提供了直观的内存分配视觉表示，适用于不同的模型配置和分布式训练策略。

### 主要功能

- **交互式内存可视化**：通过交互式树状图直观展示内存分配情况
- **内存分布分析**：使用柱状图和比例视图分析各类别内存使用情况
- **分布式训练策略**：配置张量并行、流水线并行等分布策略
- **实时内存估计**：参数变更时获得即时内存使用反馈
- **双语支持**：完整的中英文界面支持
- **配置文件上传**：导入Galvatron配置文件以进行精确的内存分析

### 内存类别

该可视化工具分析并显示以下几个类别的内存使用情况：

- **激活内存（Activation Memory）**：前向传播过程中存储激活值所使用的内存
- **模型状态（Model States）**：参数、梯度和优化器状态的总内存
  - **参数内存（Parameter Memory）**：存储模型参数所使用的内存
  - **梯度内存（Gradient Memory）**：反向传播过程中梯度所使用的内存
  - **优化器内存（Optimizer Memory）**：优化器状态所使用的内存
  - **梯度累积（Gradient Accumulation）**：多步更新中梯度累积所使用的内存

### 安装说明

1. 克隆仓库
```bash
git clone https://github.com/PKU-DAIR/Hetu-Galvatron.git
cd Hetu-Galvatron/galvatron-visualizer
```

2. 安装依赖
```bash
npm install
```

3. 启动开发服务器
```bash
npm start
```

4. 打开 [http://localhost:3000](http://localhost:3000) 查看应用

### 使用指南

1. **选择配置**：选择预定义模型或上传配置文件
2. **调整参数**：在配置面板中修改模型参数
3. **查看内存分析**：在树状图可视化中观察内存分配
4. **分析分布**：使用柱状图和比例视图了解内存使用模式
5. **探索不同策略**：测试不同的并行策略以优化内存使用

### 许可证

本项目是北京大学Hetu-Galvatron框架的一部分。更多信息，请访问[GitHub仓库](https://github.com/PKU-DAIR/Hetu-Galvatron)或阅读相关[研究论文](https://arxiv.org/abs/2211.13878)。
