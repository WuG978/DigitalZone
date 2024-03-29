---
{"dg-publish":true,"permalink":"/200-Study/Computer Organization/第 8 章 CPU 的结构和功能/","noteIcon":""}
---


# 第 8 章 CPU 的结构和功能

# 8.1 CPU 的结构

## 8.1.1 CPU 的功能

1. 控制器的功能：取指令、分析指令、执行指令并发出各种操作命令、控制程序输入及结果输出、总线管理、处理异常情况和特殊请求
2. 运算器的功能：实现算术运算和逻辑运算

综上，CPU 需要有指令控制、操作控制、时间控制、处理中断、数据加工 5 大功能。

## 8.1.2 CPU 结构框图

![CO_8_01](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_01.png)

## 8.1.3 CPU 的寄存器分类

1. 按照对用户是否可见
    1. 用户可见寄存器，用户在编写程序过程中可以读取和使用寄存器的值
        1. 通用寄存器，存放操作数，可作某种寻址方式所需的专用寄存器
        2. 数据寄存器，存放操作数（满足各种数据类型），两个寄存器拼接存放双倍字长数据
        3. 地址寄存器，存放地址，其位数应满足最大的地址范围，用于特殊的寻址方式，如段基值、栈指针
        4. 条件码寄存器，存放条件码，可作程序分支的依据，如正、负、零、溢出、进位等
    2. 不可见寄存器
2. 控制和状态寄存器
    1. 控制寄存器

        PC → MAR → M → MDR → IR

        其中 MAR、MDR、IR 用户不可见，PC 用户可见

    2. 状态寄存器

        状态寄存器存放条件码，PSW 寄存器存放程序状态字

## 8.1.4 控制单元 CU 和中断系统

1. CU 产生全部指令的微操作命令序列，它有两种设计方法：
    1. 组合逻辑设计，硬连线逻辑。速度快，适用于 RISC
    2. 微程序设计，存储逻辑。适用于 CISC，参见第 4 篇
2. 中断系统

    参见 8.4 节

## 8.1.5 ALU

参见第 6 章

# 8.2 指令周期

## 8.2.1 指令周期的基本概念

1. 指令周期：取出并执行一条指令所需的全部时间

    完成一条指令需要经过取指、分析、执行三个阶段，取指和分析所需时间称为取指周期，执行所需时间所需时间称为执行周期。

    ![CO_8_02](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_02.png)

2. 每条指令的指令周期不同

    ![CO_8_03](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_03.png)

3. 具有间接寻址的指令周期

    ![CO_8_04](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_04.png)

4. 带有中断周期的指令周期

    ![CO_8_05](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_05.png)

5. 指令周期流程

    ![CO_8_06](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_06.png)

6. CPU 工作周期的标志

    CPU 访存有四种性质：取指令（取指周期）、取地址（间址周期）、存取操作数或结果（执行周期）、存程序断点（中断周期）

    ![CO_8_07](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_07.png)

    四个标志：FE - 取指周期，IND - 间址周期，EX - 执行周期，INT - 中断周期

## 8.2.2 指令周期的数据流

1. 取指周期数据流

    ![CO_8_08](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_08.png)

    1. (PC) → MAR，PC 中的地址送到 MAR
    2. (MAR) → 地址总线 → M，MAR 中的地址送到存储器中
    3. CU → 控制总线 → 存储器，CU 将操作信号送到存储器
    4. 存储器 → 数据总线 → MDR，将指令从存储器取出送到 MDR
    5. (MDR) → IR，将 MDR 中保存的指令送到 IR
    6. (PC) + 1→ PC，CU 控制程序寄存器加 1
2. 间址周期数据流

    ![CO_8_09](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_09.png)

    假设地址已经保存在 MDR 中

    1. (MDR) → MAR，MDR 中的地址数据送到 MAR 中
    2. (MAR) → 地址总线 → M，MAR 中的地址送到存储器中
    3. CU → 控制总线 → 存储器，CU 将操作信号送到存储器
    4. 存储器 → 数据总线 → MDR，将指令从存储器取出送到 MDR
3. 执行周期数据流

    不同指令的执行周期数据流不同，详见第 9 章

4. 中断周期数据流

    ![CO_8_10](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_10.png)

    CU 指定断点保存地址，断点数据保存在 PC 中，CU 给出中断服务程序入口地址并保存在 PC 中

    1. (CU) → MAR，将 CU 指定的程序断点存储地址送到 MAR
    2. (MAR) → 地址总线 → M，MAR 中的地址送到存储器中
    3. CU → 控制总线 → 存储器，CU 将操作信号送到存储器
    4. (PC) → MDR，将 PC 中保存的断点数据送到 MDR
    5. (MDR) → 数据总线 → 存储器，将 MDR 中的数据送到存储器
    6. (CU) → PC，将 CU 指定的中断服务程序入口地址保存到 PC

# 8.3 指令流水

## 8.3.1 指令流水原理

1. 指令的串行执行：指令各个阶段依次执行，各指令之间顺序执行。

    ![CO_8_11](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_11.png)

    取指阶段只会用到取指令部件，执行指令阶段只会用到执行指令部件，此时总有一个部件时空闲的。

2. 指令的二级流水：在第一条指令的执行阶段，第二条指令进行取指。

    ![CO_8_12](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_12.png)

    若取指和执行阶段时间上完全重叠，指令周期减半，速度提高 1 倍

3. 影响指令流水效率加倍的因素
    1. 执行时间 > 取指时间

        解决方法：在取指部件和执行部件之间加一个指令部件缓冲区，该缓冲区用于缓冲取指部件取出来的指令。

        ![CO_8_13](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_13.png)

    2. 条件转移指令对指令流水的影响

        必须等上条指令结束，才能确定下条指令的地址，造成时间损失

        解决方法：分支预测方法（如何进行分支预测？）

4. 指令的六级流水

    ![CO_8_14](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_14.png)

    如上图，完成一条指令需要 6 个时间单位，串行执行需要 $6 \times 9 = 54$ 个时间单位，六级流水执行需要 $6 + 8 = 14$ 个时间单位

## 8.3.2 影响流水线性能的因素

1. 结构相关：不同指令争用同一功能部件产生资源冲突

    ![CO_8_15](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_15.png)

    图中蓝色标记的指令都将访问内存，故造成资源冲突

    解决办法：

    - 停顿
    - 指令存储器和数据存储器分开
    - 指令预取技术（适用于访存周期短的情况）
2. 数据相关：不同指令因重叠操作，可能改变操作数的读/写访问顺序
    1. 先写后读相关（RAW）

        ![CO_8_16](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_16.png)

    2. 先读后写相关（WAR）

        ![CO_8_17](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_17.png)

    3. 先写后写（WAW）

        ![CO_8_18](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_18.png)

    解决方法：

    - 后推法：遇到数据相关时，就停顿后继指令的运行，直至前面的指令的结果已经生成。
    - 采用旁路技术：又称定向技术或相关专用通路技术。其主要思想是不必待某条指令的执行结果送回到寄存器后，再从寄存器中取出该结果，作为下一条指令的源操作数，而是直接将执行结果送到其它指令所需要的地方。
3. 控制相关：由转移指令引起，条件转移指令必须等待条件判断的结果才能判断出是转移还是顺序执行。

## 8.3.3 流水线性能

流水线性能指标

- 吞吐率：单位时间内流水线所完成指令或输出结果的数量

    设 m 段的流水线各段时间为 $\Delta t$

    - 最大吞吐率：流水线在连续流动达到稳定状态后所获得的吞吐率

        $$
        T_{pmax} = {1 \over \Delta t}
        $$

    - 实际吞吐率：流水线完成 n 条指令的实际吞吐率

        $$
        T_p = {n \over {m \cdot \Delta t + (n-1) \cdot \Delta t}}
        $$

- 加速比 $S_p$

    m 段的流水线的速度与等功能的非流水线的速度之比

    设流水线各段时间为 $\Delta t$

    完成 n 条指令在 m 段流水线上共需

    $$
    T = m \cdot \Delta t + (n - 1) \cdot \Delta t
    $$

    完成 n 条指令在等效的非流水线上共需

    $$
    T' = nm \cdot \Delta t
    $$

    则

    $$
    S_p = \frac {nm \cdot \Delta t} {m \cdot \Delta t + (n - 1) \cdot \Delta t} = \frac {nm}{m + n - 1}
    $$

- 效率

    流水线中各功能段的利用率。由于流水线有建立时间和排空时间，因此各功能段的设备不能一直处于工作状态。

    $$
    \begin{align*}
    效率 &= \frac {流水线各段处于工作时间的时空区}{流水线中各段总的时空区} \\
        &= \frac {mn \Delta t}{m(m + n - 1) \Delta t} \\
        &= \frac {n}{m + n - 1}
    \end{align*}
    $$

    ![CO_8_19](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_19.png)

    效率即为上图占用面积比长方形面积

## 8.3.4 流水线的多发技术

1. 超标量技术：在每个时钟周期内可同时并发多条独立指令，即以并行操作方式将两条或两条以上指令编译并执行。

    特点：

    - 每个时钟周期内可并发多条独立指令，需要配置多个功能部件
    - 不能调整指令的执行顺序，但可通过编译优化技术，把可并行执行的指令搭配起来

![CO_8_20](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_20.png)

1. 超流水线技术：将一些流水线寄存器插入到流水线段中，即将流水线再分段。

    特点：

    - 在一个时钟周期内再分段，使一个功能部件使用多次
    - 不能调整指令执行顺序顺序，但可编译程序解决优化问题

    ![CO_8_21](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_21.png)

2. 超长指令字技术：采用多条指令在多个处理部件中并行处理的体系结构，在一个时钟周期内能流出多条指令。

    特点：

    - 由编译程序挖掘出指令间潜在的并行性，将多条能并行操作的指令组合称一条，具有多个操作码字段的超长指令字
    - 采用多个处理部件

![CO_8_22](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_22.png)

## 8.3.5 流水线结构

1. 指令流水线结构：指令级的流水技术，将指令的整个执行过程用流水线进行分段处理。

    完成一条指令分 6 段，每段需要一个时钟周期

    ![CO_8_23](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_23.png)

    若流水线不出现断流，1 个时钟周期出现 1 个结果；不采用流水线技术的情况下，6 个时钟周期出现 1 结果。理想情况下，6 级流水的速度是不采用流水技术的 6 倍。

2. 运算流水线：部件级的流水技术

    例如，完成浮点加减运算可分为对阶、尾数求和、规格化三段，分段原则：每段操作时间尽量一致

    ![CO_8_24](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_24.png)

# 8.4 中断系统

## 8.4.1 概述

1. 引发中断的各种因素
    1. 人为设置的中断：如，转管指令
    2. 程序性事故：溢出、操作码不能识别、除法非法
    3. 硬件故障
    4. I/O 设备
    5. 外部设备：用键盘中断现行程序
2. 中断系统需解决的问题
    1. 各中断源如何向 CPU 提出中断请求
    2. 当多个中断源同时提出中断请求时，中断系统如何确定优先响应哪个中断源的请求
    3. CPU 在什么条件、什么时间、以什么方式来响应中断
    4. CPU 响应中断后如何保护现场
    5. CPU 响应中断后，如何停止原程序的执行而转入中断服务程序的入口地址
    6. 中断处理结束后，CPU 如何恢复现场，如何返回到原程序的间断处
    7. 在中断处理过程中又出现了新的中断请求，CPU 该如何处理

    用软件 + 硬件来解决上述问题

## 8.4.2 中断请求标记和中断判优逻辑

1. 中断请求标记 INTR

    一个请求源设置一个 INTR 中断请求标记触发器，多个 INTR 组成中断请求标记寄存器

    ![CO_8_25](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_25.png)

    INTR 既可以分散在各个中断源的接口电路，也可以集中在 CPU 的中断系统内

2. 中断判优逻辑
    1. 硬件实现（排队器）
        1. 分散在各个中断源的接口电路中，即链式排队器（参见第 5 章）
        2. 集中在 CPU 内

        ![CO_8_26](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_26.png)

    2. 软件实现（程序查询）

        ![CO_8_27](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_27.png)

## 8.4.3 中断服务程序的入口地址寻找

1. 硬件向量法

    ![CO_8_28](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_28.png)

2. 软件查询法

    ![CO_8_29](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_29.png)

## 8.4.4 中断响应

1. 响应中断的条件：允许中断触发器 EINT = 1
2. 响应中断的时间：指令执行周期结束时刻由 CPU 发查询信号

    ![CO_8_30](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_30.png)

3. 中断隐指令

    CPU 响应中断后，即进入中断周期。在中断周期内，CPU 要自动完成一系列操作，这些就需要中断隐指令来完成。

    1. 保护程序断点

        断点存于特定地址或者栈内

    2. 寻找中断服务程序的入口地址

        向量地址 → PC（硬件向量法）

        中断识别程序入口地址 M → PC（软件查询法）

    3. 硬件关中断

        ![CO_8_31](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_31.png)

## 8.4.5 保护现场和恢复现场

1. **保护现场**：保存断点（中断隐指令完成）和寄存器内容（中断服务程序完成）
2. **恢复现场**，由中断服务程序完成

**中断服务程序工作内容：**

![CO_8_32](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_32.png)

## 8.4.6 中断屏蔽技术

1. 多重中断的概念

    当 CPU 正在执行某个中断服务程序时，另一个中断源又提出了新的中断请求，而 CPU 又响应了这个新的请求，暂时停止正在运行的服务程序，转去执行新的中断服务程序，这称为多重中断，又称中断嵌套。若 CPU 对新的请求不予响应，待执行完当前的服务程序后再响应，即为单重中断。

    ![CO_8_33](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_33.png)

2. 实现多重中断的条件
    1. 提前设置开中断指令
    2. 优先级别高的中断源有权中断优先级低的中断源

    ![CO_8_34](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_34.png)

3. 屏蔽技术
    1. 屏蔽触发器 MASK 的作用

        MASK = 0（未屏蔽），INTR 能被置 1，即可以接收中断请求并进入排队

        MASK = 1（屏蔽），INTP = 0，不能被排队选中

        ![CO_8_35](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_35.png)

    2. 屏蔽字：所有屏蔽触发器组合在一起构成一个屏蔽寄存器，屏蔽寄存器的内容称为屏蔽字

        屏蔽字与中断源的优先级是一一对应的

        ![CO_8_36](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_36.png)

    3. 屏蔽技术可改变处理优先等级

        优先级包含响应优先级和处理优先级。响应优先级次序是硬件设置好的，不可改变；处理优先级可通过重新设置屏蔽字改变。

        ![CO_8_37](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_37.png)

    4. 屏蔽技术的其它作用：可以人为地屏蔽某个中断源的请求
    5. 新屏蔽字的设置

        保护现场 → 设置屏蔽字 → 开中断 → 中断服务 → 关中断 → 恢复现场 → 恢复屏蔽字 → 开中断 → 中断返回

4. 多重中断的断点保护

    断点进栈或存入特定地址（如“0”地址），中断隐指令完成

    中断周期内执行操作：0 → MAR，CU 命令存储器写，PC → MDR（将 PC 中断点送到 MDR），(MDR) → 存储器

    多重中断程序断点都存入 0 地址的断点保护：

    ![CO_8_38](https://image-1256466424.cos.accelerate.myqcloud.com/CO_8_38.png)
