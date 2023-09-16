FROM ubuntu

WORKDIR /
ADD https://zenodo.org/record/8278363/files/RR.zip?download=1 ./

RUN apt-get update && apt install unzip

RUN unzip RR.zip -d ./


RUN cp ./RR/README.pdf ./

RUN apt install -y  python3-pip
RUN chmod 777 ./RR/QCTMC/*

RUN pip3 install psutil -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install sympy -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple