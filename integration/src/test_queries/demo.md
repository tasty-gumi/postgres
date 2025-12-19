以下是符合要求的 **Markdown格式内容**（含LaTeX数学公式），你可直接复制到文本编辑器保存为 `.md` 文件下载使用。

# 本周组会报告及Lero模型的Listwise训练方案设计
## 一、本周工作核心总结
本周核心工作是完成**Lero二元比较模型（Pairwise）的全流程训练**，针对行列引擎查询优化场景中的计划排序问题，构建了基于两两计划对比的模型训练框架，并取得初步训练效果，同时暴露了当前方案局限性，为后续Listwise方案优化奠定基础。

### 1. 训练核心细节
1.  **数据处理**
    以查询计划组为单位，组内两两生成计划对（$i \neq j$），通过`AnalyzeJsonParser`解析计划JSON提取特征，严格过滤空特征无效样本，最终生成**5170个有效训练样本**。标签依据计划延迟生成：若计划$i$延迟$\geq$计划$j$，标签设为1.0；反之设为0.0，用于标识计划$j$更优。
2.  **模型与环境配置**
    - 特征维度：基于首个有效样本确定输入特征维度为53；
    - 模型结构：初始化适配该维度的LeroNet；
    - 硬件与优化器：单GPU训练，采用Adam优化器（学习率$10^{-4}$）；
    - 损失与激活：使用BCELoss损失函数，搭配Sigmoid激活函数输出二元比较结果。
3.  **训练执行结果**
    配置100轮训练、批次大小64，训练过程中损失逐步收敛：从初始0.6567降至最终0.4829。模型成功学习到两组计划间的延迟相对关系，最终将模型权重保存至`lero_pairwise_model.pth`，整体训练耗时约3.04小时（10947.41秒）。

### 2. 现有Pairwise方案局限性
1.  **排序一致性风险**：仅学习局部两两关系，易出现传递性矛盾（如$p_1>p_2$、$p_2>p_3$但模型判断$p_3>p_1$）；
2.  **推理效率低下**：输出仅为二元判断，行列引擎实际选择最优计划时，需额外通过多次两两比对排序，增加推理延迟；
3.  **样本冗余**：组内$K$个计划会生成$K \times (K-1)$个样本，数据预处理和训练阶段存在大量冗余开销。

## 二、Lero Listwise模型训练方案
为解决Pairwise方案的缺陷，设计Listwise训练方案：模型直接接收一组固定数量的候选计划作为输入，学习组内全局相对关系，最终输出与输入等长的最优概率数组。具体设计如下：

### 1. 方案核心目标
| 输入                                                 | 输出                                                       | 约束条件                                                           |
| ---------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------ |
| 一组固定数量为$K$的候选查询计划$\{P_1,P_2,...,P_K\}$ | 长度为$K$的概率数组$\{\hat{p}_1,\hat{p}_2,...,\hat{p}_K\}$ | $\sum_{i=1}^K \hat{p}_i = 1$，$\hat{p}_i$表示$P_i$为最优计划的概率 |

### 2. 步骤1：数据预处理（适配Listwise格式）
需保留组内计划的完整性，处理流程如下：
1.  **固定组内计划数量$K$**
    - 若组内计划数$M=K$：直接保留；
    - 若$M>K$：筛选延迟覆盖均匀的$K$个计划（避免样本偏斜）；
    - 若$M<K$：用无效特征填充，后续模型中通过掩码机制过滤填充影响。
2.  **生成目标概率标签**
    基于计划延迟构建目标概率分布$\{q_1,q_2,...,q_K\}$，延迟越小，目标概率越高。采用**延迟负相关的Softmax归一化**生成标签，公式如下：
    $$
    q_i = \frac{\exp\left(-\frac{lat_i}{T}\right)}{\sum_{j=1}^K \exp\left(-\frac{lat_j}{T}\right)}
    $$
    其中：
    - $lat_i$为计划$P_i$的实际延迟；
    - $T$为温度系数（控制概率平滑度，$T$越小，最优计划的目标概率越集中）。
    示例：若延迟为$[1,2,3]$，取$T=1$，计算得：
    $$
    q_1=\frac{\exp(-1)}{\exp(-1)+\exp(-2)+\exp(-3)} \approx 0.665, \quad q_2 \approx 0.244, \quad q_3 \approx 0.091
    $$
    近似用户期望的$[0.6,0.3,0.1]$。
3.  **样本格式**
    最终训练样本格式为$(\{X_1,X_2,...,X_K\},\{q_1,q_2,...,q_K\})$，其中$X_i$为计划$P_i$的53维特征向量。

### 3. 步骤2：模型结构调整
复用原LeroNet的特征提取能力，仅调整输入层、输出层及中间适配层，结构如下：
```
输入（B×K×53）→ 共享编码器 → 独立分数映射层 → Softmax层 → 输出（B×K）
```
各模块详细说明：
1.  **共享编码器**
    复用原LeroNet核心网络作为共享编码器$f$，对每个计划的特征向量$X_i$编码，输出高维语义特征：
    $$
    h_i = f(X_i), \quad h_i \in \mathbb{R}^D
    $$
    其中$D$为编码后特征维度，共享编码器保证所有计划的特征提取逻辑一致。
2.  **独立分数映射层**
    通过全连接层$g$将编码特征映射为排序分数，确保每个计划的分数仅由自身特征决定：
    $$
    s_i = g(h_i) = W \cdot h_i + b
    $$
    其中$W \in \mathbb{R}^{1 \times D}$为权重矩阵，$b$为偏置项。
3.  **Softmax概率转换层**
    将组内所有计划的分数转换为概率分布，输出预测概率：
    $$
    \hat{p}_i = \frac{\exp(s_i)}{\sum_{j=1}^K \exp(s_j)}
    $$

### 4. 步骤3：损失函数与数学定义
选用**交叉熵损失**衡量预测分布与目标分布的差异，具体定义如下：
1.  **单组样本损失**
    引入极小值$\epsilon=10^{-8}$避免$\log(0)$错误：
    $$
    L_{single} = -\sum_{i=1}^K q_i \cdot \log(\hat{p}_i + \epsilon)
    $$
2.  **批次损失**
    设批次包含$B$个样本，批次总损失为单组样本损失的均值：
    $$
    L_{batch} = \frac{1}{B} \sum_{b=1}^B L_{single,b}
    $$

### 5. 完整训练流程（代码实现）
```python
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset

# 全局配置
CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")
GPU_LIST = [0] if CUDA else []

# 定义Listwise-LeroNet
class ListwiseLeroNet(nn.Module):
    def __init__(self, input_feature_dim=53, hidden_dim=128, K=5):
        super().__init__()
        self.K = K
        # 共享编码器（复用原LeroNet核心结构）
        self.encoder = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        # 独立分数映射层
        self.score_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, batch_plans):
        # batch_plans: [B, K, 53]
        B = batch_plans.shape[0]
        # 展平为[B*K, 53]进行编码
        flatten_plans = batch_plans.reshape(-1, batch_plans.shape[-1])
        h = self.encoder(flatten_plans)  # [B*K, hidden_dim//2]
        scores = self.score_head(h).reshape(B, self.K)  # [B, K]
        return scores

# 数据预处理函数（生成Listwise样本）
def generate_listwise_samples(query_groups, K=5, T=1):
    ajp = AnalyzeJsonParser(normalizer=None, input_relations=[])
    samples = []
    for group in query_groups:
        plans = group['plans']
        latencies = group['latencies']
        M = len(plans)

        # 调整组内计划数量为K
        if M > K:
            # 筛选延迟覆盖均匀的K个计划
            sorted_idx = np.argsort(latencies)
            step = max(1, len(sorted_idx) // K)
            selected_idx = sorted_idx[::step][:K]
            selected_plans = [plans[i] for i in selected_idx]
            selected_lats = [latencies[i] for i in selected_idx]
        elif M < K:
            # 填充无效计划（特征全0）
            selected_plans = plans + [{'plan_json': {}}] * (K - M)
            selected_lats = latencies + [np.inf] * (K - M)
        else:
            selected_plans = plans
            selected_lats = latencies

        # 提取特征并生成目标概率
        X_list = []
        valid_flag = True
        for plan in selected_plans:
            try:
                sam = ajp.extract_feature(plan['plan_json'])
                feat = sam.get_feature() if hasattr(sam, 'get_feature') else [0]*53
                if len(feat) != 53:
                    feat = [0]*53
                X_list.append(feat)
            except:
                X_list.append([0]*53)
        X = np.array(X_list, dtype=np.float32)

        # 计算目标概率
        lat_array = np.array(selected_lats, dtype=np.float32)
        exp_vals = np.exp(-lat_array / T)
        q = exp_vals / exp_vals.sum()
        samples.append((X, q))
    return samples

# 自定义数据集
class ListwiseDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

# 训练主流程
if __name__ == "__main__":
    # 1. 生成Listwise样本
    query_groups = []  # 替换为你的查询计划组数据
    K = 5
    samples = generate_listwise_samples(query_groups, K=K)
    if not samples:
        raise ValueError("No valid Listwise samples")
    
    # 2. 初始化模型与配置
    model = ListwiseLeroNet(input_feature_dim=53, K=K).to(device)
    if CUDA:
        model = nn.DataParallel(model, device_ids=GPU_LIST)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    cross_entropy_loss = nn.CrossEntropyLoss()

    # 3. 构建数据加载器
    dataset = ListwiseDataset(samples)
    batch_size = 64 * len(GPU_LIST) if CUDA else 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 4. 训练循环
    epochs = 100
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        valid_batches = 0
        for X, q in dataloader:
            X = X.to(device)
            q = q.to(device)

            # 前向传播
            scores = model(X)
            preds = torch.softmax(scores, dim=1)

            # 计算损失
            loss = cross_entropy_loss(preds, q)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            valid_batches += 1
        
        avg_loss = total_loss / valid_batches
        print(f"Epoch {epoch}, training loss: {avg_loss:.6f}")

    # 5. 保存模型
    class LeroModelListwise:
        def __init__(self, model=None):
            self.model = model
        def save(self, path):
            if isinstance(self.model, nn.DataParallel):
                torch.save(self.model.module.state_dict(), path)
            else:
                torch.save(self.model.state_dict(), path)
    lero_model = LeroModelListwise(model=model)
    lero_model.save("lero_listwise_model.pth")
```

## 三、Listwise更适合行列引擎查询优化的原因
行列引擎的查询优化核心是**从候选计划集合中一次性选择最优方案**，Listwise方案的特性与该场景高度契合，具体优势如下：
1.  **解决排序一致性问题**
    Pairwise通过两两比对推导全局排序时，易出现传递性矛盾。Listwise直接学习组内所有计划的全局关系，从根源上保证排序逻辑一致，避免行列引擎执行计划时出现逻辑冲突。
2.  **贴合实际决策流程**
    行列引擎需快速从候选集合中选定最优计划，Listwise直接输出每个计划的最优概率，可直接选取概率最高的计划执行，无需额外排序步骤；而Pairwise需多次推理比对，增加查询优化延迟。
3.  **捕捉全局依赖特征**
    行列引擎中部分计划的优劣依赖组内其他计划的特性（如某并行计划的优势仅在其他计划为串行时凸显）。Pairwise仅关注局部两两特征，无法捕捉此类全局依赖；Listwise处理整组计划时，可间接学习到计划间的协同依赖，提升复杂场景适配能力。
4.  **降低样本冗余与训练开销**
    一组$K$个计划中，Pairwise生成$K \times (K-1)$个冗余样本，而Listwise仅生成1个样本。对于行列引擎中多计划查询场景，Listwise能减少数据预处理成本和训练开销，同时避免冗余样本导致的过拟合。

---

## 保存为MD文件的步骤
1.  新建一个文本编辑器（如记事本、VS Code等）；
2.  复制上述全部内容粘贴到编辑器中；
3.  点击“保存”，文件格式选择“所有文件”，文件名后缀设为 `.md`（如 `Lero_Listwise_Report.md`）；
4.  保存完成后即可获得可下载的Markdown文件。