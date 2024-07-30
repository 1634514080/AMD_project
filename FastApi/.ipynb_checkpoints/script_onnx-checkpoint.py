from itertools import chain

import numpy as np
from bertviz import model_view
from fastapi import FastAPI, File, UploadFile, Header
from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertForSequenceClassification, BertConfig
from fastapi.middleware.cors import CORSMiddleware
import re
import json
import mimetypes
import torch
import handle
import info
from collections import OrderedDict
import predict
import document
import utils
from configparser import ConfigParser


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


from transformers.onnx.features import FeaturesManager
onnx_config = FeaturesManager._SUPPORTED_MODEL_TYPE['bert']['sequence-classification'](config)
dummy_inputs = onnx_config.generate_dummy_inputs(tokenizer1, framework='pt')
output_onnx_path="./model1.onnx"
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
options = SessionOptions()  # initialize session options
options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

# 这里的路径传上一节保存的onnx模型地址
session = InferenceSession(
    "./model1.onnx", sess_options=options, providers=["CPUExecutionProvider"]
)

# disable session.run() fallback mechanism, it prevents for a reset of the execution provider
session.disable_fallback()

model1.load_state_dict(torch.load(
    'models/Roberta.pt', map_location=torch.device('cpu')), strict=False)
model1.eval()


sentence="我是一名程序员"
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

# # 运行
# # 这里的logits要有export的时候output_names相对应
# session.run(output_names=['logits'], input_feed=inputs)
#
# outputs = session.run(output_names=['logits'], input_feed=inputs)


with torch.no_grad():
    outputs = model1(input_ids.unsqueeze(
        0), attention_mask=attention_mask.unsqueeze(0),token_type_ids=token_type_ids.unsqueeze(0))
    attention=outputs.attentions
    tokens=tokenizer1.convert_ids_to_tokens(encoding['input_ids'][0])

    model_view(attention, tokens)  # Display model view
    logits = outputs.logits
print(logits)
print(torch.max(logits, dim=1))
_, predicted_label = torch.max(logits, dim=1)
print(_)
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
print(predicted_logits)
print(np.argmax(predicted_logits, axis=1))
# 计算预测的标签
predicted_label = np.argmax(predicted_logits, axis=1)

# 打印输出结果
print("ONNX Runtime Inference Output:")
print("Predicted Label:", predicted_label)




#
#
# tokenizer2 = BertTokenizer.from_pretrained(model_name)
# config2 = BertConfig.from_pretrained(model_name)
# config2.num_hidden_layers = 2
# config2.num_labels = 11
# model2 = BertForSequenceClassification.from_pretrained(
#     model_name, config=config2)
# model2.load_state_dict(torch.load(
#     'models/Job.pt', map_location=torch.device('cpu')), strict=False)
# model2.eval()
#
#
# # 加载 NER
# tokenizer3 = AutoTokenizer.from_pretrained(model_name)
# pretrained3 = AutoModel.from_pretrained(model_name)
#
#
# # 定义下游模型
# class Model(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.tuning = False
#         self.pretrained = None
#
#         self.rnn = torch.nn.GRU(768, 768, batch_first=True)
#         self.fc = torch.nn.Linear(768, 8)
#
#     def forward(self, inputs):
#         if self.tuning:
#             out = self.pretrained(**inputs).last_hidden_state
#         else:
#             with torch.no_grad():
#                 out = pretrained3(**inputs).last_hidden_state
#
#         out, _ = self.rnn(out)
#         out = self.fc(out).softmax(dim=2)
#         return out
#
#
# model3 = torch.load('models/Ner.pt', map_location=torch.device('cpu'))
# model3.eval()
#
#
# config = ConfigParser()
# config.read('secrets.txt')
#
# app = FastAPI()
# app.secret_key = config.get('key', 'jwt_key')
#
# # 添加 CORS 中间件
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=['*'],
#     allow_credentials=True,
#     allow_methods=['*'],
#     allow_headers=['*'],
# )
#
#
# async def analysis(sentences):
#     # sentences = list(OrderedDict.fromkeys(sentences))
#     # sequence 为最初分类序列 经过上下文纠正算法得到最终分类结果 data
#     sequence = []
#     data = [[] for _ in range(7)]
#     for s in sentences:
#         p = predict.label_predict(s, tokenizer1, model1)
#         s = re.sub(r'[\s\t]{2,}', ' ', s.strip())
#         sequence.append([s, p])
#
#     # 上下文纠正算法 纠正特殊标签6 这里有两种可能 6,2->6,1 或 6,2->2,2
#     for i in range(0, len(sequence)-1):
#         if sequence[i][1] == 6 and sequence[i+1][1] == 2:
#             job_obj_patter = '|'.join(info.job_obj_keywords())
#             matches = re.findall(job_obj_patter, sequence[i][0])
#             if matches:
#                 # 6,2->6,1
#                 sequence[i+1][1] = 1
#             else:
#                 # 6,2->2,2
#                 sequence[i][1] = 2
#
#     # 至此分类完成
#     for s in sequence:
#         data[s[1]].append(s[0])
#     # 对基本信息单独NER+正则匹配
#     basic_data = {
#         'name': '',
#         'birth': '',
#         'age': 0,
#         'tel': '',
#         'email': '',
#         'college': [],
#         'loc': [],
#         'edu': ''
#     }
#
#     tag = {
#         'edu_tag': [],
#         'loc_tag': '',
#         'experience_tag': [],
#         'ability': [],
#         'total_work_time': ''
#     }
#
#     total_data = {
#         'basic_data': basic_data,
#         'job_obj': data[1],
#         'experience': data[2],
#         'award': list(OrderedDict.fromkeys(data[3])),
#         'ability': list(OrderedDict.fromkeys(data[4])),
#         'job_fit': [],
#         'tag': tag,
#         'score': 0,
#         'custom_content': {
#             'money_obj': '',
#             'self_desc': [],
#             'self_tag': []
#         }
#     }
#     handle.handle_basedata(data, basic_data, total_data, tokenizer3, model3)
#     handle.handle_job_obj(total_data)
#     handle.handle_experience(total_data, tokenizer3, model3)
#     handle.handle_ability(total_data)
#     handle.handle_job_fit(total_data, tokenizer2, model2)
#
#     return json.dumps(total_data, ensure_ascii=False)
#
#
# def R(data, code, msg):
#     mp = {}
#     if code == 200:
#         mp['res'] = data
#         mp['code'] = 200
#         mp['msg'] = msg
#     elif code == 400:
#         mp['code'] = 400
#         mp['msg'] = msg
#     elif code == 402:
#         mp['code'] = 402
#         mp['msg'] = msg
#     return mp
#
#
# get_content = {
#     'application/pdf': {
#         'func': document.get_pdf_content,
#         'err_msg': 'pdf文件解析失败'
#     },
#     'application/vnd.openxmlformats-officedocument.wordprocessingml.document': {
#         'func': document.get_docx_content,
#         'err_msg': 'docx文件解析失败'
#     },
#     'text/plain': {
#         'func': document.get_txt_content,
#         'err_msg': 'txt文件解析失败'
#     }
# }
#
#
# @app.post('/qes/')
# async def qes(Authorization: str = Header(...), file: UploadFile = File(...)):
#     mp = {}
#     if utils.verify_jwt(Authorization, app.secret_key):
#         file_type = mimetypes.guess_type(file.filename)[0]
#         sentences = []
#         if file_type in get_content:
#             try:
#                 sentences = await get_content[file_type]['func'](file)
#             except Exception as e:
#                 print(e)
#                 mp = R(None, 400, get_content[file_type]['err_msg'])
#         else:
#             mp = R(None, 400, f'Oops 请检查文件类型')
#         try:
#             res = await analysis(sentences)
#             mp = R(res, 200, '简历解析成功')
#         except Exception as e:
#             print(e)
#             mp = R(None, 400, '解析出错')
#     else:
#         mp = R(None, 402, 'token 验证失败')
#     return mp
#
#
# @app.get('/authorize/')
# async def authorize():
#     return {'code': '200', 'token': utils.create_jwt(app.secret_key)}
#
#
# # uvicorn script:app --reload
# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host='0.0.0.0', port=8000, reload=False)
