"""
损失函数：
        损失函数也叫成本函数，目标函数，代价函数，误差函数，就是用来衡量  模型好坏（模型拟合情况）的
        分类和回归问题

        分类问题：
            多分类交叉熵损失函数    CrossEntropyLoss
            二分类交叉熵损失函数    BCELoss
        回归问题：
            MAE: Mean Absolute Error,平均绝对误差
            MSE: Mean Squared Error,均方误差
            SmoothL1: 结合上述两个的特点做的升级，优化

        1、多分类交叉熵损失函数:CrossEntropyLoss
            设计思路:
                Loss = -求和 ylog(S(f(x)))
                x: 样本
                f(x):  加权求和
                S(f(x)):  处理后的概率
                y:  样本x属于某一个类别的  真实概率

            损失函数结果 = 确类别概率的对数最小化

            CrossEntropyLoss: = Softmax() + 损失计算，如果用这个损失函数，就不需额外调用 softmax() 函数

        2、二分类交叉熵损失函数:BCELoss
            公式:
                Loss = -ylog(预测值)-(1-y)log(1-预测值)

                因为公式中没有包含 Sigmoid 函数，需要手动指定 Sigmoid

        3、回归任务的损失函数
            MAE:    Mean Absolute Error,平均绝对误差
                误差的绝对值之和 / 样本总数
                类似于L1正则化，权重可以降为0，数据会变得稀松

                弊端：
                    在0点不平滑，可能错过最小值

            MSE:    Mean Squared Error,均方误差
                公式：
                    误差平方和 / 样本总数
                弊端：
                    如果差值过大，可能存在梯度爆炸的情况

            Smooth L1:
                就是基于MAE和MSE的综合，在[-1,1]是L2(MSE),其他段为L1
                这样既解决了L1不平滑的问题
                又解决了L2的梯度爆炸的问题

"""