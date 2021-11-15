#问题在118--132行，test和cauchy_stress相等，但sym_eig得到的特征值和特征向量却不相等
# 2D
# 2021.11.9
# damage
import taichi as ti
import numpy as np
ti.init(arch=ti.gpu)
steps = 25
dim = 2
max_particles = 100000
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
p_rho = 1e6
# damage有关的参数
E, nu = 4e3, 0.45  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
    (1 + nu) * (1 - 2 * nu))  # Lame parameters
dt = 1e-4
particles = ti.Struct.field({
    "d": ti.f32,
    "vol": ti.f32,
    "position": ti.types.vector(2, ti.f32),
    "velocity": ti.types.vector(2, ti.f32),
    "F": ti.types.matrix(2, 2, ti.f32),
    "C": ti.types.matrix(2, 2, ti.f32),
    "mass": ti.f32,
    "J": ti.f32,
    "lable_mark": ti.i32, # 标记应该施加什么样的力
}, shape=max_particles)
node = ti.Struct.field({
    "node_m": ti.f32,
    "node_v": ti.types.vector(2, ti.f32),
    "node_d": ti.f32,
    "laplacian_d": ti.f32,
    "node_mark":ti.i32,
}, shape=(n_grid, n_grid))
particle_num = ti.field(ti.i32, shape=())
node_pos = ti.Vector.field(2, ti.f32, shape=(n_grid, n_grid))
horizontal_stretch = ti.Vector.field(2, dtype=float, shape=())  # 水平拉伸
cauchy_stress = ti.Matrix.field(2, 2, ti.f64, shape=())

# ***************** init *************************
@ti.kernel
def add_cuboid(position: ti.template(), length: float, height: float):
    for i in range(4800):
        pos = [0.005 * (i // (height * 200)) + position[0], 0.005 * (i % (height * 200)) + position[1]]
        # pos = [ti.random() * length + position[0], ti.random() * height + position[1]]
        # print("i:", i, pos)
        particles[i].position = pos
        # 左右施加力
        if pos[0] <= position[0] + length * 0.1:
            particles[i].lable_mark = 1  # left
        elif pos[0] >= position[0] + length * 0.9:
            particles[i].lable_mark = 2  # right
        else:
            particles[i].lable_mark = 0

        # residual
        particles[i].d = 0.0
        # 粒子体积
        particles[i].vol = (dx * 0.5) ** 2
        #
        particles[i].velocity = ti.Matrix([0.0, 0.0])
        particles[i].F = ti.Matrix([[1, 0], [0, 1]])
        particles[i].C = ti.Matrix([[1, 0], [0, 1]])
        particles[i].mass = p_rho * (dx * 0.5) ** 2
        particles[i].J = 1.0
    particle_num[None] += 4800

@ti.kernel
def add_force():
    for p in range(max_particles):
        base = (particles[p].position * inv_dx - 0.5).cast(int)
        # 左侧粒子，向左的力
        if particles[p].lable_mark == 1:
            node[base].node_mark = 1
        # 右侧粒子，向右的力
        if particles[p].lable_mark == 2:
            node[base].node_mark = 2


def init():
    add_cuboid(ti.Vector([0.2, 0.3]), 0.6, 0.2)

# *************************************************


# 更新 d
@ti.kernel
def update_d():
    for p in range(particle_num[None]):
        xp = particles[p].position / dx
        base = int(xp - 0.5)
        fx = xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

        # ++++++++ QR  sign convention +++++++++++
        Q, R = QR2(particles[p].F)
        # sign convention
        r00 = R[0, 0]
        if r00 < 0:
            Q *= -1
            R *= -1

        # compute J
        particles[p].J = 1.0
        for r in ti.static(range(2)):
            particles[p].J *= R[r, r]
        # compute stress
        pk1 = calculate_pk1(Q, R, particles[p].J)

        T1 = ti.Matrix.zero(float, 2, 2)
        T2 = ti.Matrix.zero(float, 4, 2)
        sym_eig_values = ti.Vector.zero(float, 2)
        sym_eig_vector = ti.Matrix.zero(float, 2, 2)

        ########################test和cauchy_stress相等，但sym_eig得到的特征值和特征向量却不相等#############################
        # 直接给出
        test = ti.Matrix([[0.014860193, 0.0], [0.0, 0.014860193]])
        cauchy_stress = pk1 @ particles[p].F / particles[p].J
        if all(test == cauchy_stress):
            # 计算特征值和特征矩阵
            if all(cauchy_stress.transpose() == cauchy_stress):
                sym_eig_values, sym_eig_vector = ti.sym_eig(cauchy_stress, ti.f32)
                print("cauchy_stress:", cauchy_stress, "eigen_values:", sym_eig_values, "eigen_vector:", sym_eig_vector)

                test_eigen_values, test_eigen_vector =ti.sym_eig(test, ti.f32)
                print("test:", cauchy_stress, "eigen_values:", test_eigen_values, "eigen_vector:", test_eigen_vector)
            else:
                T1, T2 = ti.eig(cauchy_stress, ti.f32)


# ****************** 弹性部分 ***********************
# 1、P2G
# 2、网格操作：处理边界条件，计算网格速度
# 3、G2P
@ti.kernel
def Elastic_P2G():
    for i, j in ti.ndrange(n_grid, n_grid):
        node[i, j].node_v = [0, 0]
        node[i, j].node_m = 0
    #P2G
    for p in range(particle_num[None]):  # Particle state update and scatter to grid (P2G)
        base = (particles[p].position * inv_dx - 0.5).cast(int)
        fx = particles[p].position * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        particles[p].F = (ti.Matrix.identity(float, 2) + dt * particles[p].C) @ particles[p].F  # deformation gradient update
        # ++++++++ QR_decomposition  +++++++++++
        Q, R = QR2(particles[p].F)
        r00 = R[0, 0]
        if r00 < 0:
            Q *= -1
            R *= -1

        # compute J
        particles[p].J = 1.0
        for r in ti.static(range(2)):
            particles[p].J *= R[r, r]

        # compute stress
        pk1 = calculate_pk1(Q, R, particles[p].J)
        stress = - 4 * inv_dx * inv_dx * particles[p].vol * pk1 @ particles[p].F.transpose()
        affine = ti.Matrix.identity(float, 2)
        affine = stress + particles[p].mass * particles[p].C
        # ***********************
        for i, j in ti.static(ti.ndrange(
                3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            node[base + offset].node_v += weight * (particles[p].mass * particles[p].velocity + affine @ dpos)
            node[base + offset].node_m += weight * particles[p].mass

# QR分解
@ti.func
def QR2(Mat: ti.template()):  # 2x2 mat, Gram–Schmidt Orthogonalization
    c0 = ti.Vector([Mat[0, 0], Mat[1, 0]])
    c1 = ti.Vector([Mat[0, 1], Mat[1, 1]])
    r11 = c0.norm(1e-6)
    q0 = c0 / r11
    r12 = c1.dot(q0)
    q1 = c1 - r12 * q0
    r22 = q1.norm(1e-6)
    q1 /= r22
    Q = ti.Matrix.cols([q0, q1])
    R = ti.Matrix([[r11, r12], [0, r22]])
    return Q, R

# compute PK1
@ti.func
def calculate_pk1(q: ti.template(), r: ti.template(), J: ti.template()) -> ti.template():
    dPsiHatdR = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
    pk = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
    A = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
    # 计算 ψ 对 R 求导
    if J >= 1:
        #  lambda term contribution
        for i in ti.static(range(2)):
            dPsiHatdR[i, i] += lambda_0 * (J - 1) * J / r[i, i]
            # print("dpsi:",dPsiHatdR[i, i],"la:",lambda_0,"J:",J,"r[i,i]:",r[i,i],"J-1:",(J-1))
    else:
        for i in ti.static(range(2)):
            dPsiHatdR[i, i] += lambda_0 * (J - 1) * J / r[i, i]
    # mu term contribution
    for i in ti.static(range(2)):
        dPsiHatdR[i, i] -= mu_0 * J / r[i, i]
    for i, j in ti.static(ti.ndrange(2, 2)):
        if i <= j:
            dPsiHatdR[i, j] +=  mu_0 * r[i, j]
    # 计算 A
    A = dPsiHatdR @ r.transpose()
    for i, j in ti.static(ti.ndrange(2, 2)):
        if i > j:
            A[i, j] = A[j, i]
    # compute P
    pk = q @ A @ (r.inverse()).transpose()
    # print(pk)
    return pk

@ti.kernel
def Elastic_grid_operator():
    # grid normolization
    for i, j in ti.ndrange(n_grid, n_grid):
        if node[i, j].node_m > 0:  # No need for epsilon here
            node[i, j].node_v = (1 / node[i, j].node_m) * node[i, j].node_v  # Momentum to velocity
            # horizontal_stretch
            if node[i, j].node_mark == 1:
                node[i, j].node_v += dt * horizontal_stretch[None]
            elif node[i, j].node_mark == 2:
                node[i, j].node_v -= dt * horizontal_stretch[None]
            if i < 3 and node[i, j].node_v[0] < 0: node[i, j].node_v[0] = 0  # Boundary conditions
            if i > n_grid - 3 and node[i, j].node_v[0] > 0: node[i, j].node_v[0] = 0
            if j < 3 and node[i, j].node_v[1] < 0: node[i, j].node_v[1] = 0
            if j > n_grid - 3 and node[i, j].node_v[1] > 0: node[i, j].node_v[1] = 0

@ti.kernel
def Elastic_G2P():
    # G2P
    for p in range(particle_num[None]):  # grid to particle (G2P)
        base = (particles[p].position * inv_dx - 0.5).cast(int)
        fx = particles[p].position * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(
                3, 3)):  # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = node[base + ti.Vector([i, j])].node_v
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        particles[p].velocity, particles[p].C = new_v, new_C
        particles[p].position += dt * particles[p].velocity  # advection
# **************************************************


gui = ti.GUI("test_elastic", res=512, background_color=0xFFFFFF)
horizontal_stretch[None] = [-50, 0]
init()
while gui.running:
    for s in range(steps):
        add_force()
        update_d()
        Elastic_P2G()
        Elastic_grid_operator()
        Elastic_G2P()
    gui.circles(particles.position.to_numpy(), color=0x0000CD, radius=1.5)
    gui.show()
