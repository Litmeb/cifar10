# cifar10
<p>一个cifar10，可以识别10种物体</p>
<p>环境是pytorch2.8.0,python3.9,cuda12.9</p>
<p>用pytorch做的一个基于cnn的一个小模型，网络结构是抄的：</p>
<img src="https://p.sda1.dev/27/6b557608adcd54e85bd8793d79e66035/image.png">
<p>对验证集的正确率变化图：</p>
<img src="https://p.sda1.dev/27/1fa3c50467b911728cc55c114fb02515/image.png" >
<p>最终正确率接近70%，但对训练集正确率有近90%，过拟合比较严重，还有优化的空间</p>
<p>损失函数定义为交叉熵，对验证集测试的loss变化图：</p>
<img src="https://p.sda1.dev/27/51d916b0c678e812823c90e935ea2017/image.png">
<p>可以看出尚未饱和，还能进一步训练降低loss</p>
