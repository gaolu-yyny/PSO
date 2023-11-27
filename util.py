import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation

# 以下的arrow_length表示的是一个比例，可设置为[0.0,1.0]
#      dpi表示绘制的GIF的画面质量，如需提高质量，可设置为300
#      frames_per_iter表示将每次迭代的画面分割为多少帧，比如你迭代50次，那么生成的GIF图片就由 50*10=500张画面组成
def plot_animation(pso, objective_func, title, xlim=(-100, 100), ylim=(-100, 100), 
                        arrow_length=None, dpi=100, frames_per_iter=10):
    """
    绘制粒子群算法的优化动画。

    Parameters:
    --------------------
    X_list : list
        粒子位置的列表。
    V_list : list
        粒子速度的列表。
    objective_func : function
        目标函数。
    title : str
        图表标题。
    xlim : tuple, optional
        x 轴范围，默认为 (-100, 100)。
    ylim : tuple, optional
        y 轴范围，默认为 (-100, 100)。
    arrow_length : float, optional
        箭头长度，默认为None，可设置为 0.1。
    dpi : int, optional
        图片分辨率，每英寸点数，默认为 100。
    frames_per_iter : int, optional
        每次迭代中的帧数，默认为 10。

    Returns:
    ----------------------
    animation : FuncAnimation
        动画对象。
    """

    if not pso.record_mode:
        print("错误提示: record_mode 未启用。不能创建动画。\n\t你应该使用 pso.record_mode = True # 开启记录详细值功能")
        return
    
    X_list, V_list = pso.record_value['X'], pso.record_value['V']

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定使用的中文字体
    plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

    # 创建画布和坐标轴
    fig, ax = plt.subplots(1, 1, dpi=dpi) # python 3.5 不支持 dpi 参数，可以考虑figsize参数
    # fig, ax = plt.subplots(1, 1)
    ax.set_title(title, loc='center')
    line = ax.plot([], [], 'b.')

    # 创建目标函数的等高线图
    # np.linspace(start, stop, num) 会生成 num 个在 start 到 stop 范围内均匀分布的数。在这里，xlim 和 ylim 分别表示 x 和 y 轴的范围，而 40 表示生成的坐标点数量。
    X_grid, Y_grid = np.meshgrid(np.linspace(xlim[0], xlim[1], 40), np.linspace(ylim[0], ylim[1], 40))
    Z_grid = objective_func((X_grid, Y_grid))
    contour = ax.contour(X_grid, Y_grid, Z_grid, 20) # 20 表示生成 20 条等高线。
    
    # 添加等高线标签
    ax.clabel(contour, inline=True, fontsize=8)

    # 设置坐标轴范围
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # 绘制一个圆作为约束条件
    # t = np.linspace(0, 2 * np.pi, 40)
    # ax.plot(0.5 * np.cos(t) + 1, 0.5 * np.sin(t), color='r')

    # 显示画布
    plt.ion()
    plt.show()

    def update_scatter(frame):
        """
        更新动画帧的散点图。
        """
        # 计算迭代次数和每次迭代的进度
        i, j = frame // frames_per_iter, frame % frames_per_iter

        # 设置标题
        ax.set_title('迭代次数 = {}'.format(i))

        # 计算粒子位置
        X_tmp = X_list[i] + V_list[i] * j / 10.0

        # 绘制箭头
        if arrow_length is not None:            
            for x, v in zip(X_tmp, V_list[i]):
                ax.arrow(x[0], x[1], arrow_length * v[0], arrow_length * v[1], head_width=0.05, head_length=0.05, fc='r', ec='r')

        # 更新动画点的位置
        plt.setp(line, 'xdata', X_tmp[:, 0], 'ydata', X_tmp[:, 1]) # python 3.5 不支持

        # 返回更新后的元素
        return line

    # 创建动画并返回
    animation = FuncAnimation(fig, update_scatter, blit=True, interval=25, frames=len(X_list) * frames_per_iter)
    return animation

def plot_meshgrid(objective_func,title, xlim=(-100, 100), ylim=(-100, 100),dpi=100):
    ''' 绘制等高线图，绘制等高线,使用示例如下：
    # x1_lim=(-2.048-0.6, 2.048+0.6) # 将x轴扩展0.6
    # x2_lim=(-2.048-0.6, 2.048+0.6) # 将y轴扩展0.6
    # plot_meshgrid(objective_function, "函数1的等高线", xlim=x1_lim, ylim=x2_lim, dpi=300)
    '''

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定使用的中文字体
    plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

    # 创建画布和坐标轴
    fig, ax = plt.subplots(1, 1, dpi=dpi)
    ax.set_title(title, loc='center')

    # 创建目标函数的等高线图
    # np.linspace(start, stop, num) 会生成 num 个在 start 到 stop 范围内均匀分布的数。在这里，xlim 和 ylim 分别表示 x 和 y 轴的范围，而 40 表示生成的坐标点数量。
    X_grid, Y_grid = np.meshgrid(np.linspace(xlim[0], xlim[1], 40), np.linspace(ylim[0], ylim[1], 40))
    Z_grid = objective_func((X_grid, Y_grid))
    contour = ax.contour(X_grid, Y_grid, Z_grid, 10) # 20 表示生成 20 条等高线。

    # 添加等高线标签
    ax.clabel(contour, inline=True, fontsize=8)

    # 设置坐标轴范围
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # 显示画布
    plt.ion()
    plt.show(block=True)

def plot_3d_surface(objective_func, title, xlim=(-100, 100), ylim=(-100, 100), dpi=100):
    '''绘制二元目标函数的三维图像, 使用示例如下：
    # x1_lim=(-2.048-0.6, 2.048+0.6) # 将x轴扩展0.6
    # x2_lim=(-2.048-0.6, 2.048+0.6) # 将y轴扩展0.6
    # plot_meshgrid(objective_function, "函数1的等高线", xlim=x1_lim, ylim=x2_lim, dpi=300)
    '''

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建画布和坐标轴
    # fig = plt.figure(figsize=(6, 6), dpi=dpi)    
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # 创建目标函数的网格
    x = np.linspace(xlim[0], xlim[1], 150)
    y = np.linspace(ylim[0], ylim[1], 150)
    X, Y = np.meshgrid(x, y)
    Z = objective_func((X, Y))

    # 绘制三维曲面图
    surface = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow', linewidth=0, antialiased=False)

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)

    # 创建独立坐标轴用于放置颜色条
    # 颜色条的左边界位于图形宽度的80%处，底边界位于图形高度的15%处，宽度为图形宽度的2%，高度为图形高度的70%
    cax = fig.add_axes([0.80, 0.15, 0.02, 0.7])

    # 添加颜色条
    fig.colorbar(surface, cax=cax, shrink=0.5, aspect=10)

    # 显示画布
    plt.ion()
    plt.show(block=True)