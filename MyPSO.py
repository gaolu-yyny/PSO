import numpy as np
import math

# PSO 算法在这个实现中是用于求解目标函数的最小值。如果你的问题是最大化问题，你可以通过修改目标函数来转换为相应的最小化问题。
class MyPSO:
    def __init__(self, func, dim=None, pop=40, max_iter=150, lb=-1e5, ub=1e5, w=0.8, c1=0.5, c2=0.5,
                 verbose=False, enable_learning_factors=False):
        
        # 初始化 PSO 类
        self.func = self.func_transformer(func)  # 转换输入的函数为向量化形式
        self.w = w  # 惯性权重
        self.set_weight_mode(weight_mode = "linear") ;
        self.cp, self.cg = c1, c2  # 控制个体最佳和全局最佳的参数
        self.pop = pop  # 粒子数
        self.dim = dim  # 粒子的维度，即 func 的变量数
        self.max_iter = max_iter  # 最大迭代次数
        self.verbose = verbose  # 是否打印每次迭代的结果

        # 设置粒子位置的上下限
        self.lb, self.ub = np.array(lb) * np.ones(self.dim), np.array(ub) * np.ones(self.dim)
        assert self.dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        # 初始化粒子位置、速度和适应度值
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.dim))
        v_high = self.ub - self.lb
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.dim))  # 粒子的速度
        self.Y = self.cal_y()  # y = f(x) for all particles
        self.pbest_x = self.X.copy()  # 每个粒子的个体最佳位置历史
        self.pbest_y = np.array([[-np.inf]] * pop)  # 每个粒子的个体最佳目标值历史
        self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # 所有粒子的全局最佳位置
        self.gbest_y = -np.inf  # 所有粒子的全局最佳目标值
        self.gbest_y_hist = []  # 每次迭代的 gbest_y
        self.update_gbest()

        
        self.record_mode = False # 默认不开启记录详细值功能
        # 记录详细值
        self.record_value = {'X': [], 'V': [], 'Y': []}
        # 默认不打印每次迭代的结果
        self.verbose = False
        # 记录当前迭代次数
        self.iter_num = 0

        # 用于控制是否启用学习因子的调整方式
        self.enable_learning_factors = enable_learning_factors
        if enable_learning_factors:
            self.set_learning_factors()

    def set_weight_mode(self, weight_mode='linear', inertia_weight=None, constriction_factor=None, w_start=0.9, w_end=None,
                        w_max=None, w_min=None, random_delta=None):
        """
        设置权重调整方式。

        Parameters:
        --------------------
        weight_mode : str, optional
            权重的调整方式，可选值为 'linear'、'adaptive'、'random'、'constriction'，默认为 'linear'。
        w : float, optional
            惯性权重，仅在 'linear' 模式下使用，表示初始惯性权重， 默认为 None。
        constriction_factor : float, optional
            收缩因子，仅在 'constriction' 模式下使用，表示惯性权重的缩小因子，默认为 None。
        w_start : float, optional
            线性递减模式下的初始权重，仅在 'linear_decline' 模式下使用，默认为 0.9。
        w_end : float, optional
            线性递减模式下的终止权重，仅在 'linear_decline' 模式下使用，默认为 None。
        w_max : float, optional
            自适应调整权重时的最大权重，默认为 None。
        w_min : float, optional
            自适应调整权重时的最小权重，默认为 None。
        random_delta: float, optional
            随机权重模式下的权重变化范围，默认为 None。
        
        see to: https://blog.csdn.net/qq_41053605/article/details/89156125
        """
        self.weight_mode = weight_mode
        self.inertia_weight = inertia_weight
        self.constriction_factor = constriction_factor
        self.w_start = w_start
        self.w_end = w_end
        self.w_max = w_max
        self.w_min = w_min        
        self.random_delta = random_delta

    # def set_learning_factors(self, c1_start=3, c1_end=1, c2_start=1, c2_end=3):
    #     '''
    #     设置学习因子的调整方式。
    #
    #     Parameters:
    #     --------------------
    #     c1_start : float, optional
    #         初始个体最佳学习因子，默认为 2。
    #     c1_end : float, optional
    #         终止个体最佳学习因子，默认为 0.5。
    #     c2_start : float, optional
    #         初始全局最佳学习因子，默认为 0.5。
    #     c2_end : float, optional
    #         终止全局最佳学习因子，默认为 2。
    #     '''
    #     self.c1_start, self.c1_end = c1_start, c1_end
    #     self.c2_start, self.c2_end = c2_start, c2_end
    #     self.cp = c1_start
    #     self.cg = c2_start

    def set_learning_factors(self, c1_start=2, c1_end=0.5, c2_start=2, c2_end=0.5):
        # 设置学习因子的初始和结束值
        self.c1_start = c1_start
        self.c1_end = c1_end
        self.c2_start = c2_start
        self.c2_end = c2_end

    def set_learning_factors_mode(self, mode='linear', **params):
        self.learning_factor_mode = mode

        # 使用传入的参数或默认值设置学习因子
        self.set_learning_factors(
            c1_start=params.get('c1_start', 2),
            c1_end=params.get('c1_end', 0.5),
            c2_start=params.get('c2_start', 2),
            c2_end=params.get('c2_end', 0.5)
        )

    def update_learning_factors(self, iter_num):
        if self.learning_factor_mode == 'synchronous':
            cmax = self.c1_start
            cmin = self.c1_end
            self.cp = cmax - (cmax - cmin) * iter_num / self.max_iter
            self.cg = self.cp
        elif self.learning_factor_mode == 'asynchronous':
            cstart1 = self.c1_start
            cend1 = self.c1_end
            cstart2 = self.c2_start
            cend2 = self.c2_end
            self.cp = cstart1 + (cend1 - cstart1) * iter_num / self.max_iter
            self.cg = cstart2 + (cend2 - cstart2) * iter_num / self.max_iter
        if self.learning_factor_mode == 'constriction':
            # 设置个体和社会学习因子的标准值
            phi1 = self.c1_start  # 个体学习因子
            phi2 = self.c2_start # 社会学习因子
            phi = phi1 + phi2
            global kappa
            kappa = 2 / abs(2 - phi - np.sqrt(phi ** 2 - 4 * phi))



    # def update_learning_factors(self, iter_num):
    #     '''
    #     更新学习因子。
    #     初始使用较大的c1值和较小的c2值，增加多样性，后期让c1的值线性降低，c2的值线性增加，增强粒子收敛能力。
    #
    #     Parameters:
    #     --------------------
    #     iter_num : int
    #         当前迭代次数。
    #     '''
    #     total_iter = self.max_iter
    #     if iter_num <= total_iter:
    #         # 根据线性降低的方式更新学习因子
    #         self.cp = self.c1_start - (self.c1_start - self.c1_end) * iter_num / total_iter
    #         self.cg = self.c2_start + (self.c2_end - self.c2_start) * iter_num / total_iter
    #     else:
    #         # 在达到最大迭代次数后保持不变
    #         self.cp = self.c1_end
    #         self.cg = self.c2_end

    def update_V(self):
        '''更新每个粒子速度'''
        every_particle_w = self.cal_every_particle_w() # 根据设定好的权重模式计算每一个粒子的权重
        if self.enable_learning_factors:
            self.update_learning_factors(self.iter_num)  # 更新学习因子

        # 更新粒子速度
        if self.set_learning_factors_mode=='constriction':
            self.V = kappa * (every_particle_w[:, np.newaxis] * self.V +
                            self.cp * np.random.rand(self.pop, self.dim) * (self.pbest_x - self.X) +
                            self.cg * np.random.rand(self.pop, self.dim) * (self.gbest_x - self.X))

        else:
            self.V = every_particle_w[:, np.newaxis] * self.V + \
                    self.cp * np.random.rand(self.pop, self.dim) * (self.pbest_x - self.X) + \
                    self.cg * np.random.rand(self.pop, self.dim) * (self.gbest_x - self.X)
    
    def update_X(self):
        '''更新每个粒子位置'''
        self.X = self.X + self.V
        self.X = np.clip(self.X, self.lb, self.ub)

    @staticmethod
    def func_transformer(func):
        '''
        将函数向量化的转换器
        :param func: 输入的函数
        :return: 向量化后的函数
        '''
        def func_transformed(X):
            return np.array([func(x) for x in X])

        return func_transformed

    def cal_every_particle_w(self):
        '''根据设定的权重模式计算每个粒子的权重'''
        if self.weight_mode in ['linear', 'linear_decline', 'adaptive', 'random']:
            inertia_weight = self.inertia_weight or self.w
            if self.weight_mode == 'linear':
                every_particle_w = inertia_weight * np.ones(self.pop)
            elif self.weight_mode == 'linear_decline':
                # 在 'linear_decline' 模式下使用线性递减权重
                k = len(self.gbest_y_hist)  # 当前迭代次数
                T = self.max_iter  # 最大迭代次数
                w_end = self.w_end or 0.1  # 终止权重
                w_start = self.w_start  # 初始权重
                every_particle_w = (w_end + (w_start - w_end) * (1 - k / T)) * np.ones(self.pop)
            elif self.weight_mode == 'adaptive':
                f_avg = np.mean(self.pbest_y)  # 平均适应度值
                valid_pbest_y = self.pbest_y[self.pbest_y > -np.inf]  # 排除 inf 值
                if len(valid_pbest_y) > 0:
                    f_min = np.min(valid_pbest_y)  # 最小适应度值
                    w_max = self.w_max or inertia_weight
                    w_min = self.w_min or 0.1
                    every_particle_w = np.zeros(self.pop)
                    for idx, f in enumerate(self.pbest_y):
                        if f <= f_avg:
                            # 当 f <= f_avg 时，使用相应的公式计算权重
                            every_particle_w[idx] = w_min - (w_max - w_min) * (f - f_min) / (f_avg - f_min)
                        else:
                            # 当 f > f_avg 时，使用最大权重
                            every_particle_w[idx] = w_max
                else:
                    # 如果所有个体的适应度值都是 inf，则使用初始权重
                    every_particle_w = inertia_weight * np.ones(self.pop)
                
            elif self.weight_mode == 'random':
                u_max = self.w_max or 1.0
                u_min = self.w_min or 0.1
                u = u_min + (u_max - u_min) * np.random.rand(self.pop)
                delta = self.random_delta or 0.1
                every_particle_w = u + delta * np.random.randn(self.pop)
        return every_particle_w

    def cal_y(self):
        '''计算每个粒子的适应度值'''
        self.Y = self.func(self.X).reshape(-1, 1)
        return self.Y

    def update_pbest(self):
        '''更新个体最佳位置'''
        self.need_update = self.pbest_y < self.Y # 检查每个粒子的适应度值是否优于其个体最佳适应度值

        # 根据需要更新个体最佳位置和适应度值
        self.pbest_x = np.where(self.need_update, self.X, self.pbest_x)
        self.pbest_y = np.where(self.need_update, self.Y, self.pbest_y)

    def update_gbest(self):
        '''更新全局最佳位置'''
        idx_max = self.pbest_y.argmax()  # 找到个体最佳适应度值中的最小值的索引
        if self.gbest_y < self.pbest_y[idx_max]:  # 如果全局最佳适应度值大于新找到的个体最佳适应度值
            self.gbest_x = self.X[idx_max, :].copy()  # 更新全局最佳位置为对应的个体位置
            self.gbest_y = self.pbest_y[idx_max]  # 更新全局最佳适应度值为对应的个体最佳适应度值

    def recorder(self):
        '''记录当前状态'''
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['V'].append(self.V)
        self.record_value['Y'].append(self.Y)

    def run(self, max_iter=None, precision=None, N=20, convergence_threshold=None, rel_tol=1e-8, unchanged_times=None):
        '''
        运行粒子群算法优化过程。

        Parameters:
        --------------------
        max_iter : int, optional
            最大迭代次数，默认为类属性中设置的最大迭代次数。
        precision : float, optional
            如果 precision 为 None，则运行 max_iter 步。
            如果 precision 为 float，则在连续 N 次迭代中 pbest 之间的差异小于 precision 时停止循环。
        N : int, optional
            当使用 precision 时，表示连续满足条件的最小迭代次数。
        convergence_threshold: float, optional
            收敛阈值。用于指定平均适应度的阈值。在每次迭代之后，计算当前种群的平均适应度值，如果平均适应度与设定的阈值在允许的误差范围，就提前结束迭代。
        rel_tol : flot, optional
            指定误差范围，默认为1e-8
        unchanged_times : int, optional
            极值连续不变的次数 unchanged_times ，用于判断是否结束迭代，默认为5。
            
        Returns:
        ----------------------
        best_x : array_like
            最佳位置的坐标。
        best_y : float
            最佳适应度值。

        Notes:
        ----------------------
        该方法执行粒子群算法的优化过程。通过更新粒子的速度和位置，计算适应度值，并更新个体最佳位置和全局最佳位置。
        如果使用 precision 参数，算法将在满足条件的情况下提前停止。迭代过程中，最佳适应度值和位置将记录在历史中。
        '''
        self.max_iter = max_iter or self.max_iter  # 初始化迭代计数器
        c = 0  # 追踪满足条件的连续迭代次数
        unchanged_counter = 0  # 连续不变计数器
        prev_best_y = None  # 用于存储上一次的 best_y

        # 迭代开始
        for iter_num in range(self.max_iter):
            self.iter_num += 1
            self.update_V()  # 更新粒子速度
            self.update_X()  # 更新粒子位置
            self.cal_y()  # 计算适应度值
            self.update_pbest()  # 更新个体最佳位置
            self.update_gbest()  # 更新全局最佳位置

            # 记录全局最佳适应度值
            self.gbest_y_hist.append(self.gbest_y)
            self.recorder()  # 记录当前状态
            
            # 打印迭代信息（如果 verbose 为 True）
            avg_fitness = np.mean(self.pbest_y) # 计算当前种群的平均适应度值
            if self.verbose:                
                print('迭代次数: {}, 最佳适应度: {} 位置: {} 平均适应度：{:.8}'.format(iter_num+1, self.gbest_y, self.gbest_x, avg_fitness))
                
            # 根据收敛阈值 convergence_threshold，判断是否结束迭代  
            #   根据收敛阈值，即目标函数的最小值点，作为结束迭代的条件
            #   调用方式：pso.run(convergence_threshold=-428.72)，其中为-428.72目标函数的最小值
            if convergence_threshold is not None:
                if math.isclose(avg_fitness, convergence_threshold, rel_tol=rel_tol) : 
                    break

            # 根据 precision 参数，判断是否结束迭代            
            #   当种群的适应度范围，连续N次最大适应度值 - 最小适应度值 < precision，结束迭代
            #   调用方式：pso.run(precision=1e-7, N=10)
            if precision is not None:
                # 计算当前个体最佳适应度值的最大值与最小值之差，即种群中适应度值的范围。这个范围用于度量种群的多样性。
                tor_iter = np.amax(self.pbest_y) - np.amin(self.pbest_y)
                # 判断当前适应度值的范围是否小于用户设定的精度阈值 
                if tor_iter < precision:
                    c = c + 1
                    if c > N:
                        break
                else:
                    c = 0
            
            # 根据连续不变的次数 unchanged_times ，判断是否结束迭代
            if unchanged_times is not None:
                if prev_best_y is not None and math.isclose(prev_best_y, self.gbest_y, rel_tol=rel_tol):
                    unchanged_counter += 1
                    if unchanged_counter >= unchanged_times:
                        break
                else:
                    unchanged_counter = 0               
            prev_best_y = self.gbest_y

        # 记录最终结果
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y
