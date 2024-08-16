# 使用Ubuntu 22.04 LTS基础镜像
FROM nvidia/cuda:12.1.1-base-ubuntu22.04  AS base

# 设置非交互式安装
ENV DEBIAN_FRONTEND=noninteractive

# 更新包列表并安装wget
RUN apt-get update && apt-get install -y wget vim

# 安装Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /usr/local/miniconda \
    && rm Miniconda3-latest-Linux-x86_64.sh

# 手动设置PATH环境变量
ENV PATH=/usr/local/miniconda/bin:$PATH

# 更新Conda并清理
RUN conda update -n base -c defaults conda && conda clean -a -y


FROM base AS kolors_env

# 创建Python 环境
RUN conda create -n kolors python=3.8 -y && \
    conda clean -a -y

# 激活SHELL    
SHELL ["conda", "run", "-n", "kolors", "/bin/bash", "-c"]

# 复制项目文件
COPY requirements.txt .

# 安装依赖
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple



FROM kolors_env AS kolors

WORKDIR /data/Kolors


# 激活SHELL    
SHELL ["conda", "run", "-n", "kolors", "/bin/bash", "-c"]

# 复制项目文件
COPY . .

# 安装项目
RUN python setup.py install

CMD ["/bin/bash", "-c", "source activate kolors && exec python -u manage.py"]