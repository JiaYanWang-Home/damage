Elastic_P2G()、Elastic_grid_operator()、Elastic_G2P()为了产生F变换，计算cauchy_stress
问题在update_d()，计算cauchy_stress，即使test和cauchy_stress矩阵相同，ti.sym_eig求出的特征值和特征矩阵依然不同
