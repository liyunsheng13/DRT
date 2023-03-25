echo "Please navigate to this repository folder"
echo "Preparing environment"
pip install gdown
echo "Downloading datasets"
wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip
unzip clipart.zip
unzip infograph.zip
unzip painting.zip
unzip quickdraw.zip
unzip real.zip
unzip sketch.zip
rm -rf *.zip
echo "Downloading labels"
gdown 1TL6_-1vEDYlRBxV6_nYawf6CvKhpb1Ik
unzip image_list.zip
echo "Downloading weights"
mkdir weights
cd weights
gdown 1mh1jpUWQrginSACZvZDmtyYeh-TZUxBS
gdown 16zmGRRnXwsTMgj2-RKhwWdaOLXkozXMl
gdown 15YhOjPjuutHrcK-m511OERu_4vIVYArD
gdown 1O4JwTDudqT1aj2VfFxgU1ld7bk0Hlcth
gdown 1ygMj4nJU74qywMbdq2DvQyyZZHngBD-3
gdown 1FVNy6OVkptKCL6rp7SqRlrZ5aYM-77vy
cd ..
mkdir out
echo "Running tests"
ython drt.py --batch-size 12 --num-layer 2 --save out --src_path clipart_comb.txt --trg_path quickdraw_train.txt --val_path quickdraw_test.txt --weight weights/quickdraw.pth.tar --evaluate
echo "Done"
echo "See folder out/ for details"
echo "Running training experiment"
echo "Downloading pre-trained weights"
cd weights
gdown 1xNmYXhSxNNOenSd8n87NWVtiyL5JrFXC
cd ..
echo "Training..."
python drt.py --batch-size 16 --epochs 5 --gamma 0.01 --lmbd 4.0 --lr 0.001 --lr_f 0.01 --momentum 0.9 --num-layer 2 --pretrain weights/resnet_dy_pretrained.pth --save_path out --src_path clipart_comb.txt --trg_path clipart_train.txt --val_path clipart_test.txt --schedule 5 10 15
echo "Done :)"
