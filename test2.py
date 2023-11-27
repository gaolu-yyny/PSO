import numpy as np
import matplotlib.pyplot as plt
from MyPSO import MyPSO
from util import plot_animation, plot_meshgrid,plot_3d_surface
from scipy.optimize import minimize

# 测试单个案例，并生成 GIF 动画
def test_and_generate_animation(objective_function, generate_animation=True):
    pso_init_params = {'func':objective_function, 'dim': 2,
                    'pop': 20, 'max_iter': 200, 'w': 0.8, 'c1': 2, 'c2':2,
                    'lb':[-1.024, -1.024], 'ub':[1.024, 1.024],
                    'enable_learning_factors': True}
    # 如果要启用学习因子，需要将上面的 enable_learning_factors 设置为 True，否则以下配置无效
    learning_factors_params = {'mode': 'asynchronous', 'c1_start': 2.05, 'c1_end': .5, 'c2_start': 2.05, 'c2_end': .5}

    # 初始化粒子群算法
    pso = MyPSO(**pso_init_params)  
    pso.record_mode = True # 开启记录详细值功能
    pso.verbose = True # 打印每次迭代的结果

    pso.set_learning_factors_mode(**learning_factors_params)

    # 设置权重模式
    #pso.set_weight_mode(weight_mode='linear_decline', w_max=0.9, w_end=0.4) # 设置权重调整方式为线性递减
    #pso.set_weight_mode(weight_mode='adaptive', inertia_weight=0.8, w_max=0.9, w_min=0.4) # 设置权重调整方式为自适应调整
    #pso.set_weight_mode(weight_mode='random', w_max=0.9, w_min=0.5, random_delta=0.5) # 设置权重调整方式为随机权重

    # 三种不同的结束迭代的方式
    #pso.run(convergence_threshold=-4.7, rel_tol=1e-5) # 设置收敛阈值作为结束迭代的条件
    pso.run(precision=1e-7, N=10) # 当种群的适应度范围，[连续10次]最大适应度值 - 最小适应度值 < precision，结束迭代
    #pso.run(unchanged_times=5, rel_tol=1e-7) # 设置极值5次未改变作为结束迭代的条件

    #non_negative_array = [x if x >= 0 else -x for x in pso.gbest_y_hist]
    # 打印找到的最优解
    print('最优解 (x) 是:', pso.gbest_x, '最大值 (y) 是:', pso.gbest_y)

    # 绘制优化过程中全局最优解的变化
    plt.rcParams['font.sans-serif']= ['Arial']# 指定使用的中文字体
    plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号
    plt.plot(pso.gbest_y_hist)
    plt.xlabel('迭代次数'), plt.ylabel('适应度值'),plt.title('适应度进化曲线')
    plt.show(block=True)

    # 根据你参入的 generate_animation 决定是否打印 GIF 动画，并且只打印迭代次数小于50的
    if generate_animation and len(pso.gbest_y_hist) < 50:
        # 绘制动画
        x1_lim=(-1.024-0.2, 1.024+0.2) # 将GIF动画的x轴扩展0.2
        x2_lim=(-1.024-0.2, 1.024+0.2) # 将GIF动画的y轴扩展0.2
        animation = plot_animation(pso, objective_function, '粒子群算法优化-多极值', x1_lim, x2_lim, dpi=100,frames_per_iter=5)
        # 保存动画为GIF文件
        animation.save('粒子群算法优化-多极值.gif', writer='pillow')

# 注意：这个函数不具备可移植性
def run_pso_with_stats(pso_init_params, learning_factors_params, weight_modes, total_runs_per_mode=100):
    '''
    运行 PSO 算法并进行统计

    Parameters:
    --------------------
    weight_modes : list
        不同权重模式的列表，每个元素是一个包含权重设置参数的字典。
    total_runs_per_mode : int, optional
        每种权重模式运行的总次数，默认为100。
    '''
    for weight_mode_params in weight_modes:
        success_count = 0
        failure_count = 0
        successful_iterations = []

        for _ in range(total_runs_per_mode):
            pso = MyPSO(**pso_init_params)
            pso.record_mode = True
            pso.verbose = False  # 如果要查看每次运行的详细信息，请设置为 True

            # 使用 “**” 语法将 weight_mode_params 字典中的键值对作为关键字参数传递给 set_weight_mode 方法。这样做的效果相当于将字典中的每个键值对解包成关键字参数。
            pso.set_weight_mode(**weight_mode_params)
            pso.set_learning_factors(**learning_factors_params)

            # 运行 PSO 算法
            # pso.run(convergence_threshold=-4.7, rel_tol=1e-9) # 设置收敛阈值作为结束迭代的条件
            # pso.run(precision=1e-7, N=10) # 当种群的适应度范围，[连续20次]最大适应度值 - 最小适应度值 < precision，结束迭代
            pso.run(unchanged_times=5, rel_tol=1e-8) # 设置极值5次未改变作为结束迭代的条件

            # 检查是否成功（达到收敛）
            if pso.iter_num < pso_init_params['max_iter']:
                success_count += 1
                successful_iterations.append(pso.iter_num)
            else:
                failure_count += 1

        success_rate = success_count / total_runs_per_mode if total_runs_per_mode > 0 else 0
        average_successful_iterations = np.mean(successful_iterations) if success_count > 0 else 0

        if pso_init_params['enable_learning_factors']:
            print("学习因子参数：{}".format(learning_factors_params))
        print("权重模式参数: {}".format(weight_mode_params))
        print("成功次数: {}".format(success_count))
        print("失败次数: {}".format(failure_count))
        print("成功率: {:.2f} %".format(success_rate * 100))
        print("平均成功迭代次数: {:.2f}".format(average_successful_iterations))
        print("\n")

# 使用 scipy.optimize 模块中的 minimize 函数来求解这个目标函数的极小值
# 注意：这个函数也是求解的局部最优，你可以通过改变初始猜测值 initial_guess = [0, 0] 进行验证
def optimize(objective_function):
    ''' 使用 minimize 函数求解目标函数的最优解 '''
    bounds = [(-1.024, 1.024), (-1.024, 1.024)] # 设置变量的取值范围    
    initial_guess = [0, 0] # 设置变量的初始猜测值    
    max_iterations = 5000 # 设置最大迭代次数
    result = minimize(objective_function, initial_guess, bounds=bounds, options={'maxiter': max_iterations})
    # 输出结果
    print("最优解：", result.x) # 最优解： [-1.024 -1.024] 或 [-1.024 1.024]
    print("最优目标值：", result.fun) # 最优目标值： 0.9438597560626838

if __name__ == "__main__":

    def objective_function(x):
        x1, x2 = x
        return (4 - (x1**2 + 2*x2**2 - 0.3*np.cos(3*np.pi*x1) - 0.4*np.cos(4*np.pi*x2)))
    
    # 说明：以下总共提供了①~③个功能，虽其并不互斥，但建议每次只进行其中一个功能，以便观测控制台输出
    # 举例：想要测试功能③时，请将功能①、功能②的代码进行注释。

    #---------------------------- 功能① ----------------------------
    # 利用 python 中的scipy.optimize计算最优解，你可以取消下行注释来求目标函数的局部最优解
    # optimize(objective_function) 

    #---------------------------- 功能② ----------------------------
    # 进行单独测试 PSO 算法，你可以取消下列注释，进行PSO算法的测试。
    #   - 如需调节 PSO 算法参数，请调节test_and_generate_animation中的pso_init_params（见第9行）
    #   - 如果你想切换不同的权重模式，可采用 pso.set_weight_mode 方法（见第23~26行，注释信息为“# 设置权重模式”处）
    #   - 如果你需要设置不同的终止迭代的方式，可在调用 pso.run 方法时，传入不同的参数实现（见第28~31行，注释信息为“# 两种不同的结束迭代的方式”）
    # np.set_printoptions(suppress=True, precision=8) # 在需要打印的地方设置numpy打印选项
    test_and_generate_animation(objective_function, generate_animation=False) # 进行单独测试 PSO 算法

    #---------------------------- 功能③ ----------------------------
    # 运行不同权重模式的 PSO 算法并进行统计

    # PSO 算法的初始化参数    
    # 建议w 取 [0.4, 0.9]；c1,c2 取 [0,4]
    pso_init_params = {'func':objective_function, 'dim': 2,
                        'pop': 50, 'max_iter': 200, 'w': 0.8, 'c1': 2, 'c2':2,
                        'lb':[-1.024, -1.024], 'ub':[1.024, 1.024],
                        'enable_learning_factors': True}         

    # 如果要启用学习因子，需要将上面的 enable_learning_factors 设置为 True，否则以下配置无效
    learning_factors_params = {'mode': 'asynchronous', 'c1_start': 2.05, 'c1_end': .5, 'c2_start': 2.05, 'c2_end': .5}

    weight_modes = [
        {'weight_mode': 'linear', 'inertia_weight': 0.8},
        {'weight_mode': 'linear_decline', 'w_max': 0.9, 'w_end': 0.4},
        {'weight_mode': 'adaptive', 'inertia_weight': 0.8, 'w_max': 0.9, 'w_min': 0.4},
        {'weight_mode': 'random', 'w_max': 0.9, 'w_min': 0.4, 'random_delta': 0.5}
    ]

    # 以下代码将执行上面 weight_modes 配置的四种模式，如果你只需要执行其中的任意几种模式，你可以将其余不想执行的模式所在行进行注释即可
    #   - 如果你只想执行其中一种模式，那么你可以取消上面功能②的注释，并注释下面的代码
    # run_pso_with_stats(pso_init_params, learning_factors_params, weight_modes, total_runs_per_mode=200)

    #---------------------------- 功能④ ----------------------------
    # 绘制等高线
    x1_lim=(-1.024-0.2, 1.024+0.2) # 将x轴扩展0.2
    x2_lim=(-1.024-0.2, 1.024+0.2) # 将y轴扩展0.2    
    # plot_3d_surface(objective_function, '目标函数2的三维图像', xlim=x1_lim, ylim=x2_lim, dpi=300)
    # plot_meshgrid(objective_function, "函数2的等高线", x1_lim, x2_lim, dpi=300)