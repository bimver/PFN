# PFN
PFN is a saliency model built on CNNs(VGG-16,Resnet or DenseNet), this code provide a version of VGGPFN which is built on VGG-16
##Dependencies
- Caffe
- Python
- Linux

###Test your images
1. Download the project code
2. Download our pretrained model from Baidu Yun (https://pan.baidu.com/s/1K2bzeA-VY_WM4c2rwCEBDw)
3. Change the image path and save path in demo.py
4. Run --python demo.py

###Train your own model
1. Change the image path and ground-truth path in create_caffe_data.py
2. Run --python create_caffe_data.py
3. Chang the txt path(created by create_caffe_data.py) in PFN.prototxt
4. Run --python solve.py


In training, a pretrained VGG-16 model is needed, you can find it easily in the Internet.
