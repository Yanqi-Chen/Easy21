# Easy21
David Silver系列课程的[练习](www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/Easy21-Johannes.pdf)，游戏为简化版本的21点。

三部分为：

- Monte-Carlo打表$Q(s,a)$（作为后面练习中的真实$Q^\ast(s,a)$）
- Sarsa($\lambda​$)打表$Q(s,a)​$
- Sarsa($\lambda$)用线性函数逼近$Q(s,a)$，线性函数的特征$\phi(s,a)$是一个对状态空间和动作空间都分段之后组合成的一个one-hot向量

