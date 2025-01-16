
### OpenCV中的光流跟踪与S3算法中的区别如下：
1. 金字塔构建时，构建的图像不是按行连续存储的, isContinuous为false；<font color=yellow>而S3没有填充padding，超出画面的点后续不会参与计算;</font>
2. 在Optical Flow前先构建完整梯度图像；<font color=yellow>S3在跟踪过程中才实时计算52领域的梯度；</font>
3. 双线性插值计算灰度值和梯度值的权重会乘以一个较大的常数，以提高浮点数的计算精度；<font color=yellow>S3没有对权重进行缩放；</font>
4. 每一个点的光流跟踪是在一个以该点为中心的默认大小为21*21的局部窗口中来计算的，遍历该窗口中的所有点，计算光度残差，构建正规方程；<font color=yellow>而S3只计算根据偏移量得到的52个点的最小二乘；</font>
5. 点跟踪失败的排除，根据构建的Hessian矩阵是否是奇异矩阵，来判断跟踪成功与否；<font color=yellow>S3主要靠反向光流来排除；</font>
6. 迭代过程中梯度雅可比保持不变，进而Hessian矩阵不变，只更新b向量；<font color=yellow>而S3每次迭代实时计算梯度雅可比；</font>
7. 光流迭代停止条件为：达到最大迭代次数30次或者像素点位移增量小于等于0.01；<font color=yellow>而S3迭代停止条件为达到最大迭代次数5次或者增量出现NaN.</font>
8. Forward additive vs. Invers compositional
9. LK use Scharr operator calculate derivative
10. LK先按金字塔从高到低（从粗到精），再按点遍历