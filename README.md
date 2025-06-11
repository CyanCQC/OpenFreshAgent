# 🥬 FreshAgent Alpha


## 🍎 FreshAgent
FreshAgent 是**电子科技大学旸谷青年科创中心“鲜”而易见团队**开发的一款**智慧农业**领域，执行**产业级批量果蔬检测**的智能体，旨在帮助不具备果蔬检测全栈能力（操作设备、录入数据、知识检索、报告汇总）的操作人员通过自然语言完成专业级的果蔬检测任务。

FreshAgent 填补了中国国内基于大语言模型的农业智能体空缺，利用图像识别技术、光谱分析技术进行果蔬新鲜度评估与分类，适用于零售、物流、食品安全等领域，帮助实现智能化的质量控制。

FreshAgent Alpha是 FreshAgent 的基础模型，实现了基本的功能路由与函数调用。“鲜”而易见团队将模型开源，推动中国农业智能体的发展。

![img.png](figs/img.png)

## 📖 技术路线

FreshAgent 采用传统的功能路由机制集成多个功能模块。用户输入被大脑模型（Brain）拆解为多个需求，并根据需求类型调用不同的模块进行处理。最终根据用户需求进行结果汇总。

![framework.png](figs/framework.png)

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


## 🚀 快速开始

### 🌏 环境准备

1. 克隆仓库：

```bash
git clone https://github.com/yourusername/FreshAgent-Alpha.git
cd FreshAgent-Alpha
```

2. 配置环境：

### 🏃‍ 运行应用

启动服务：

```bash
uvicorn app.main:app --reload
```

访问API文档：http://127.0.0.1:8000/docs

---

## 🤝 贡献指南

欢迎贡献！请阅读[CONTRIBUTING.md](CONTRIBUTING.md)。

---

## 📄 许可证

本项目采用 [MIT许可证](LICENSE) 开源。

---

## 📬 联系方式

- **“鲜”而易见团队负责人**：      杨长
- **FreshAgent Alpha 开发总监**：王云溪 [@CyanCQC](https://github.com/CyanCQC)
- **邮箱**：                     FreshNIR@163.com

---

## 🙏 致谢

- [电子科技大学](https://www.uestc.edu.cn/)
- [旸谷青年科创中心]()