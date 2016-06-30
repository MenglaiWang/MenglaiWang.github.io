---
layout: post
title: caffe+Ubuntu14.04.10 +cuda7.0/7.5+CuDNNv4 安装
---

特别说明：
0. Caffe 官网地址：http://caffe.berkeleyvision.org/

1. 本文为作者亲自实验完成，但仅限用于学术交流使用，使用本指南造成的任何不良后果由使用者自行承担，与本文作者无关，谢谢！为保证及时更新，转载请标明出处，谢谢！

2. 本文旨在为新手提供一个参考，请高手勿要吐槽，有暴力倾向者，请绕道，谢谢！

3. 本文使用2016年3月22日下载的caffe-master版本，运行平台为：Ubuntu 14.04，CUDA7.0，cuDNN v4，Intel Parallel Studio XE Cluster 2016，OpenCV 2.4.9

4. 安装过程，因为平台不同、设备不同、操作者不同，会遇到各种奇怪的问题和报错信息，请善用Caffe官网的Issues和caffe-user论坛,以及Google和Baidu。参考本指南，请下载最新版caffe-master，新版本很多文件已经变更。

5. 最后更新时间：2016年4月14日。


本文主要参考博客
https://www.0x6f.info/index.php/Geek-Tools/Ubuntu-14-04-LTS-%E5%AE%89%E8%A3%85GeForce-GTX-650%E6%98%BE%E5%8D%A1%E9%A9%B1%E5%8A%A8.html
http://ouxinyu.github.io/Blogs/20151108001.html
http://blog.csdn.net/yaoxingfu72/article/details/45363097

## 第一部分 linux安装

我的分区设置如下：
根分区： \ 100G
Swap交换分区：8G ，这里设置为何内存一样，据说小于16G的内存，就设置成内存的1-2倍，所以这里貌似设少了。。。。
boot分区：200M
Home分区：剩余的空间，鉴于Imagenet，PASCAL VOC之类的大客户，建议500G，至少300G以上，我这里是1.9T。

## 第二部分 驱动安装

我的显卡是GTX Titan Black，属于700系列。
[先去官网下载驱动](http://www.geforce.cn/drivers)
安装依赖：

```
sudo apt-get install build-essential pkg-config xserver-xorg-dev linux-headers-`uname -r`
```

这个地方可能会出现安装不上什么`libcheese`之类的问题，我Google了很长时间，最后的解决办法就是把这乱七八糟的依赖全都装上。。。。

```
sudo apt-get install  libglew-dev libcheese7 libcheese-gtk23 libclutter-gst-2.0-0 libcogl15 libclutter-gtk-1.0-0 libclutter-1.0-0  xserver-xorg-input-all
```

**接下来需要注意** 可能有两种情况，有的人需要先重启一下，有的不需要。

接下来禁止开源驱动：

```
sudo vim /etc/modprobe.d/blacklist.conf
```
在blacklist.conf后面添加

```
blacklist vga16fb
blacklist nouveau
blacklist rivafb
blacklist nvidiafb
blacklist rivatv
```
进入命令模式：

```
Ctrl+Alt+F1
```
关闭图形环境：

```
sudo service lightdm stop
```
找到下好的run文件目录,安装

```
sudo sh ./NVIDIA-Linux-x86_64-361.28.run
```
这里因为禁止了开源驱动，所以如果安装失败就有可能跪了，但是不要着急重装系统，可以先在命令行下把之前加入黑名单的驱动那部分给注释掉，然后重启，再添加一次黑名单，重来一次，有可能就成功了。如果还不行，呵呵，重装系统吧。

```
#blacklist vga16fb
#blacklist nouveau
#blacklist rivafb
#blacklist nvidiafb
#blacklist rivatv
```

在安装的过程，需要阅读版权信息和一些相关就驱动的删除和某些模块的下载，这里全部选择同意和OK就行了，其他的不用管，直到安装完成。
启动图形环境：

```
sudo service lightdm start
```
出现Ubuntu登录界面,登录即可。

查看是否安装成功：
打开系统设置-》详细信息
看到自己的显卡信息，则表示安装完成。
![这里写图片描述](http://img.blog.csdn.net/20160414185358416)

## 第三部分

PS：特别推荐*.deb的方法，目前已提供离线版的deb文件，该方法比较简单，不需要切换到tty模式，因此不再提供原来的*.run安装方法，这里以CUDA 7.0为例。

### 一、CUDA Repository

获取CUDA安装包,安装包请自行去NVidia官网下载。（https://developer.nvidia.com/cuda-downloads）

```
$ sudo dpkg -i cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb
$ sudo apt-get update
```

### 二、CUDA Toolkit

```
$ sudo apt-get install -y cuda
```

### 三、Environment Variables

```
$ export CUDA_HOME=/usr/local/cuda-7.5
$ export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
$ PATH=${CUDA_HOME}/bin:${PATH}
$ export PATH
```
PS：如果出现安装失败，重启系统，重新安装一遍基本都可以解决，实在不行就卸载原来的驱动再安装一遍。
a. 卸载现有驱动

```
$ sudo nvidia-installer --uninstall
```
b. 重装CUDA Toolkit


### 四. cuda环境配置：

```
$ sudo vim /etc/ld.so.conf.d/cuda.conf
/usr/local/cuda/lib64
/lib
```

完成lib文件的链接操作，执行：

```
$ sudo ldconfig -v
```

### 五、CuDNN

a、安装前请去先官网下载最新的cuDNN (cudnn-7.0-linux-x64-v4.0-prod.tgz)

```
$ sudo cp include/cudnn.h /usr/local/include
$ sudo cp lib64/libcudnn.* /usr/local/lib
```
b、链接cuDNN的库文件

```
$ sudo ln -sf /usr/local/lib/libcudnn.so.4.0.7 /usr/local/lib/libcudnn.so.4
$ sudo ln -sf /usr/local/lib/libcudnn.so.4 /usr/local/lib/libcudnn.so
$ sudo ldconfig -v
```

### 六、MKL

如果没有Intel MKL或者嫌麻烦， 可以用下列命令安装免费的atlas

```
sudo apt-get install libatlas-base-dev
```
然后你就可以跳到第七步了。

这里可以选择（ATLAS，MKL或者OpenBLAS），我这里使用MKL，首先下载并安装英特尔® 数学内核库 Linux* 版MKL(Intel(R) Parallel Studio XE Cluster Edition for Linux 2016)，下载链接是：https://software.intel.com/en-us/intel-education-offerings， 使用学生身份（邮件 + 学校）下载Student版，填好各种信息，可以直接下载，同时会给你一个邮件告知序列号。下载完之后，要把文件解压到home文件夹(或直接把tar.gz文件拷贝到home文件夹,为了节省空间，安装完记得把压缩文件给删除喔～)，或者其他的ext4的文件系统中。

接下来是安装过程，先授权，然后安装：

```
$ tar zxvf parallel_studio_xe_2016.tar.gz （如果你是直接拷贝压缩文件过来的）
$ chmod a+x parallel_studio_xe_2016 -R
$ sh install_GUI.sh
```
PS: 安装的时候，建议使用root权限安装，过程中会要求输入Linux的root口令。（设置方法：命令行：$ sudo passwd）

mkl环境设置

1. 新建intel_mkl.conf，并编辑之：

```
$ sudo vim /etc/ld.so.conf.d/intel_mkl.conf
/opt/intel/lib/intel64
/opt/intel/mkl/lib/intel64
```
2. 完成lib文件的链接操作，执行：

```
$ sudo ldconfig -v
```

### 七、配置python环境

1. 其他依赖项，确保都成功

```
 sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler protobuf-c-compiler protobuf-compiler
```

2. 安装pycaffe必须的一些依赖项：

```
sudo apt-get install -y python-numpy python-scipy python-matplotlib python-sklearn python-skimage python-h5py python-protobuf python-leveldb python-networkx python-nose python-pandas python-gflags Cython ipython
```

### 八、编译Caffe

```
git clone https://github.com/BVLC/caffe
cp Makefile.config.example Makefile.config
```

配置Makefile.config文件（仅列出修改部分）
a. 启用CUDNN，去掉"#"

```
USE_CUDNN := 1
```
b. 配置mkl（如果有的话）

```
BLAS := mkl
```
c.编译

```
$ make -j8
$ make test
$ make runtest
$ make pycaffe
```
