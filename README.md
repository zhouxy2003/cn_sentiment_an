# 基于 BERT 的中文情感分类任务

<p align="center">
  <a href="https://github.com/google-research/bert">
    <img src="https://img.shields.io/badge/bert-brightgreen.svg" alt="bert">
  </a>
    <a href="https://github.com/huggingface/transformers">
    <img src="https://img.shields.io/badge/transformers-blueviolet.svg" alt="tf">
  </a>
</p>

本文档介绍了如何使用 `transformers` 库和相关工具实现情感分析任务。脚本基于预训练的 BERT 模型（`bert-base-chinese`），对文本进行分类，标签为正面（positive）、负面（negative）和中性（neutral）。

---
- `cn_sentiment.py` 是训练文件
- `tag.py` 是测试文件

## 1. **环境依赖**

在运行代码前，请确保已安装以下 Python 库：

- `transformers`
- `datasets`
- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `scikit-learn`
- `torch`

可以使用以下命令安装依赖：

```bash
pip install transformers datasets pandas numpy seaborn matplotlib scikit-learn torch
```

---

## 2. **脚本工作流程**

### **步骤 1：加载并处理数据**
1. **读取输入 CSV 文件**  
   使用 pandas 读取数据：
   ```python
   df = pd.read_csv('test.csv', encoding='gbk')
   ```
   输入 CSV 文件应包含以下两列：
   - `text`：文本内容。
   - `sentiment`：情感标签（positive、negative 或 neutral）。

2. **定义标签映射**  
   将情感标签映射为数值：
   ```python
   target_map = {'positive': 1, 'negative': 0, 'neutral': 2}
   df['target'] = df['sentiment'].map(target_map)
   ```

3. **提取文本和标签**  
   只保留文本和标签列，并保存为新的 CSV 文件：
   ```python
   df2 = df[['text', 'target']]
   df2.columns = ['sentence', 'label']
   df2.to_csv('data.csv', index=None)
   ```

---

### **步骤 2：划分数据集**
- 使用 `datasets` 库加载预处理后的数据：
  ```python
  raw_datasets = load_dataset('csv', data_files='data.csv')
  ```

- 将数据划分为训练集和测试集（测试集占比 30%）：
  ```python
  split = raw_datasets['train'].train_test_split(test_size=0.3, seed=42)
  ```

---

### **步骤 3：加载分词器**
使用预训练的 BERT 分词器对句子进行标记化处理：
```python
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

def tokenize_fn(batch):
    return tokenizer(batch['sentence'], truncation=True)

tokenized_datasets = split.map(tokenize_fn, batched=True)
```

---

### **步骤 4：加载预训练模型**
加载 `bert-base-chinese` 模型，并设置分类任务的标签数（3 个）：
```python
model = AutoModelForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=3)
```

---

### **步骤 5：设置训练参数**
定义训练参数，如训练轮次、批量大小等：
```python
training_args = TrainingArguments(
    output_dir='training_dir',  # 输出文件夹
    evaluation_strategy='epoch',  # 每个 epoch 进行验证
    save_strategy='epoch',  # 每个 epoch 保存模型
    num_train_epochs=3,  # 训练轮次
    per_device_train_batch_size=16,  # 每个设备的训练批量大小
    per_device_eval_batch_size=64  # 每个设备的验证批量大小
)
```

---

### **步骤 6：定义性能评估函数**
计算模型的准确率（accuracy）和 F1 分数：
```python
def compute_metrics(logits_and_labels):
    logits, labels = logits_and_labels
    predictions = np.argmax(logits, axis=-1)
    acc = np.mean(predictions == labels)
    f1 = f1_score(labels, predictions, average='macro')
    return {'accuracy': acc, 'f1': f1}
```

---

### **步骤 7：训练模型**
使用 `Trainer` 进行模型训练和验证：
```python
trainer = Trainer(
    model,  # 模型实例
    training_args,  # 训练参数
    train_dataset=tokenized_datasets["train"],  # 训练数据集
    eval_dataset=tokenized_datasets["test"],  # 验证数据集
    tokenizer=tokenizer,  # 分词器
    compute_metrics=compute_metrics  # 评估函数
)

trainer.train()
```

---

## 3. **脚本文件结构**
- `test.csv`：输入数据文件，包含 `text` 和 `sentiment` 两列。
- `data.csv`：预处理后的数据文件，包含 `sentence` 和 `label` 两列。
- `training_dir/`：训练输出目录，存储模型和训练日志。

---

## 4. **注意事项**
- 确保输入数据格式正确（如 CSV 文件的列名和编码）。
- 根据任务需求调整训练参数（如轮次、批量大小）。
- 输出的模型可用于后续推理任务。

--- 
