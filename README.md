# 🥬 FreshAgent Alpha


![logo.png](figs/logo.png)

## 🍎 FreshAgent
FreshAgent 是**电子科技大学旸谷青年科创中心“鲜”而易见团队**开发的一款**智慧农业**领域，执行**产业级批量果蔬检测**的智能体，旨在帮助不具备果蔬检测全栈能力（操作设备、录入数据、知识检索、报告汇总）的操作人员通过自然语言完成专业级的果蔬检测任务。

FreshAgent 填补了国内的 LLM-based 农业智能体空缺，是**首个聚焦果蔬检测的智能体**。利用图像识别、光谱分析等技术进行果蔬新鲜度评估与分类，适用于零售、物流、食品安全等领域，帮助实现智能化的质量控制。

FreshAgent Alpha是 FreshAgent 的基础模型，实现了基本的功能路由与函数调用。“鲜”而易见团队将模型向社区开源，力求以绵薄之力助力AI+农业发展。

![img.png](figs/img.png)

## 📖 技术路线

FreshAgent 采用传统的功能路由机制集成多个功能模块。用户输入被大脑模型（Brain）拆解为多个需求，并根据需求类型调用不同的模块进行处理。最终根据用户需求进行结果汇总。

![framework.png](figs/framework.png)

### 🔬 Analyze

FreshAgent 对于 data 目录下的数据具有访问能力，支持对光谱/图像进行多模态分析。要使用 data 目录进行原生数据分析，需要：

1. 创建并进入对应数据目录，强烈建议以水果名命名，便于Brain识别
```commandline
mkdir ./data/apple
cd ./data/apple
```

2. 创建 img 与 spec 目录，分别存放图像数据与光谱数据
```commandline
mkdir ./img ./spec
```

3. 上传数据。FreshAgent按去除后缀的文件名进行数据匹配


### 🧠 Brain

用户输入进入 Brain 后，Brain 进行任务拆解。

![Brain.png](figs/Brain.png)

### ❓ Query

数据库操作涉及到sql代码编写与执行。该部分对于Agent的能力至关重要。因此，FreshAgent Alpha对其进行了多项优化。

- FreshAgent Alpha 在提示词中注入数据库信息，保证模型对数据库具有实时了解
- FreshAgent Alpha 引入了 Supervisor (监管者模型)，对于代码的合法性进行检查
- 通过 Supervisor 检查的代码，才会被送入 `DatabaseHandler` 执行

![Query.png](figs/Query.png)

### 🔍 Retrieve

FreshAgent Alpha 提供对本地知识库的原生支持。用户可以通过在 LocalDataBase 目录中上传 `pdf` 或 `txt` 文件，添加本地知识。打开 FreshAgent Alpha 的增强检索开关，会自动开启知识库检索。

![Retrieval.png](figs/Retrieval.png)

## ⚙ 环境配置

1. 依赖安装
```commandline
  pip install \
  markdown \
  PyQt5 \
  apscheduler \
  zhipuai \
  openai \
  numpy \
  pandas \
  scikit-learn \
  scipy \
  joblib \
  matplotlib \
  langchain \
  pymysql \
  pdfminer.six \
  dashscope \
  pyserial \
```

2. 环境变量

以下环境变量在运行前需根据实际情况进行配置：

- `ZHIPU_API_KEY`：智谱清言 API 密钥
- `DASHSCOPE_API_KEY`：通义千问 API 密钥
- `DEEPSEEK_API_KEY`：DeepSeek API 密钥
- `DB_HOST`、`DB_USER`、`DB_PASSWORD`、`DB_NAME`、`DB_PORT`、`DB_CHARSET`：数据库连接信息
- `EMAIL_HOST`、`EMAIL_PORT`、`EMAIL_USERNAME`、`EMAIL_PASSWORD`、`EMAIL_USE_SSL`：邮件服务配置
- `AGENT_LOCATION`：报告中使用的地理位置

可在终端通过 `export` 或在 `.env` 文件中设置上述变量。

## 🚀 快速开始

### 🌏 环境准备

1. 克隆仓库

```bash
git clone https://github.com/yourusername/FreshAgent-Alpha.git
cd FreshAgent-Alpha
```

2. 配置环境

3. 运行可视化页面（基于PyQt5开发）
```bash
python GUI.py
```

### 🏃‍ 运行接口

启动服务：

```bash
python api_server.py
```

接口示例：

```bash
# 提交文本任务
curl -X POST http://127.0.0.1:8000/api/chat \
     -H "Content-Type: application/json" \
     -d '{"user_input": "数据库中的砂糖桔呈现出什么特征？", "enhanced": false}'

# 上传图片
curl -X POST http://127.0.0.1:8000/api/image \
     -F image=@/path/to/img.jpg \
     -F prompt="请分析" \
     -F enhanced=true
```
---

## 🤝 贡献指南

欢迎贡献！请[联系我们]()

---

## 📄 许可证

本项目采用 [MIT许可证](LICENSE) 开源

---

## 📬 联系方式

- **“鲜”而易见团队负责人**：      杨长
- **FreshAgent Alpha 开发总监**：王云溪 [@CyanCQC](https://github.com/CyanCQC)
- **邮箱**：                     FreshNIR@163.com

---

## 🙏 致谢

- [电子科技大学](https://www.uestc.edu.cn/)
- [旸谷青年科创中心]()