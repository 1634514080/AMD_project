from itertools import chain
import numpy as np
from bertviz import model_view
from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertForSequenceClassification, BertConfig, AutoConfig
import torch



# 加载分类器 model1 用于语句分类  | model2 用于岗位匹配
# model_name = 'hfl/chinese-roberta-wwm-ext'
model_name = './models/bert-roberta'
tokenizer1 = BertTokenizer.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name)
config.num_hidden_layers = 2
config.num_labels = 7
config.output_attentions = True
model1 = BertForSequenceClassification.from_pretrained(
    model_name, config=config)

model1.load_state_dict(torch.load(
    'models/Roberta.pt', map_location=torch.device('cpu')), strict=False)
model1.eval()

from transformers.onnx.features import FeaturesManager
onnx_config = FeaturesManager._SUPPORTED_MODEL_TYPE['bert']['sequence-classification'](config)
dummy_inputs = onnx_config.generate_dummy_inputs(tokenizer1, framework='pt')
output_onnx_path="models/Roberta.onnx"
torch.onnx.export(
    model1,
    (dummy_inputs,),
    f=output_onnx_path,
    input_names=list(onnx_config.inputs.keys()),
    output_names=list(onnx_config.outputs.keys()),
    dynamic_axes={
        name: axes for name, axes in chain(onnx_config.inputs.items(), onnx_config.outputs.items())
    },
    do_constant_folding=True,
    opset_version=onnx_config.default_onnx_opset,
)
options = SessionOptions()
options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

# 这里的路径传上一节保存的onnx模型地址
session = InferenceSession(
    "./model1.onnx", sess_options=options, providers=["CPUExecutionProvider"]
)

session.disable_fallback()


sentence="程序员"
encoding = tokenizer1.encode_plus(
        sentence,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )


input_ids = encoding['input_ids'].squeeze()
attention_mask = encoding['attention_mask'].squeeze()
token_type_ids = encoding['token_type_ids'].squeeze()

with torch.no_grad():
    outputs = model1(input_ids.unsqueeze(
        0), attention_mask=attention_mask.unsqueeze(0),token_type_ids=token_type_ids.unsqueeze(0))
    attention=outputs.attentions
    tokens=tokenizer1.convert_ids_to_tokens(encoding['input_ids'][0])
    model_view(attention, tokens)  # Display model view
    logits = outputs.logits

_, predicted_label = torch.max(logits, dim=1)
# 打印输出结果
print("Model Inference Output:")
print(predicted_label)


# 从编码中获取输入 ID 和注意力掩码
input_ids = encoding['input_ids'].squeeze().numpy()
attention_mask = encoding['attention_mask'].squeeze().numpy()

# 为 ONNX Runtime 准备输入数据，添加批次维度
input_ids_unsqueezed = input_ids[np.newaxis, :]
attention_mask_unsqueezed = attention_mask[np.newaxis, :]
# 如果模型需要 token_type_ids，生成一个全0的数组
token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

token_type_ids_unsqueezed = token_type_ids[np.newaxis, :]
# 准备输入字典
inputs = {
    session.get_inputs()[0].name: input_ids_unsqueezed,
    session.get_inputs()[1].name: attention_mask_unsqueezed,
    session.get_inputs()[2].name: token_type_ids_unsqueezed  # 假设 token_type_ids 是第三个输入
}

# 运行模型
outputs = session.run(["logits"], inputs)

# 获取预测的 logits
predicted_logits = outputs[0]
# 计算预测的标签
predicted_label = np.argmax(predicted_logits, axis=1)

# 打印输出结果
print("ONNX Runtime Inference Output:")
print("Predicted Label:", predicted_label)




#
#
tokenizer2 = BertTokenizer.from_pretrained(model_name)
config2 = BertConfig.from_pretrained(model_name)
config2.num_hidden_layers = 2
config2.num_labels = 11
config2.output_attentions = True
model2 = BertForSequenceClassification.from_pretrained(
    model_name, config=config2)
model2.load_state_dict(torch.load(
    'models/Job.pt', map_location=torch.device('cpu')), strict=False)
model2.eval()


onnx_config = FeaturesManager._SUPPORTED_MODEL_TYPE['bert']['sequence-classification'](config2)
dummy_inputs = onnx_config.generate_dummy_inputs(tokenizer2, framework='pt')
output_onnx_path="models/Job.onnx"
torch.onnx.export(
    model2,
    (dummy_inputs,),
    f=output_onnx_path,
    input_names=list(onnx_config.inputs.keys()),
    output_names=list(onnx_config.outputs.keys()),
    dynamic_axes={
        name: axes for name, axes in chain(onnx_config.inputs.items(), onnx_config.outputs.items())
    },
    do_constant_folding=True,
    opset_version=onnx_config.default_onnx_opset,
)

#
#
# 加载 NER
tokenizer3 = AutoTokenizer.from_pretrained(model_name)
pretrained3 = AutoModel.from_pretrained(model_name)
config3=AutoConfig.from_pretrained(model_name)
# 定义下游模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tuning = False
        self.pretrained = None

        self.rnn = torch.nn.GRU(768, 768, batch_first=True)
        self.fc = torch.nn.Linear(768, 8)

    def forward(self, input_ids):
        if self.tuning:
            out = self.pretrained(**input_ids).last_hidden_state
        else:
            with torch.no_grad():
                out = pretrained3(**input_ids).last_hidden_state

        out, _ = self.rnn(out)
        out = self.fc(out).softmax(dim=2)
        return out


