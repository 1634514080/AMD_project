import numpy as np
from onnxruntime import SessionOptions, GraphOptimizationLevel, InferenceSession
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertForSequenceClassification, BertConfig
import torch
from itertools import chain


def export_onnx(model_name,weight_file):
    tokenizer1 = BertTokenizer.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name)
    config.num_hidden_layers = 2
    config.num_labels = 7
    config.output_attentions = True
    model1 = BertForSequenceClassification.from_pretrained(
        model_name, config=config)

    model1.load_state_dict(torch.load(
        weight_file, map_location=torch.device('cpu')), strict=False)
    model1.eval()

    from transformers.onnx.features import FeaturesManager
    onnx_config = FeaturesManager._SUPPORTED_MODEL_TYPE['bert']['sequence-classification'](config)
    dummy_inputs = onnx_config.generate_dummy_inputs(tokenizer1, framework='pt')
    output_onnx_path = weight_file.split('.')[0]+'.onnx'
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
    return output_onnx_path

def test(model_name,weight_file,onnx_file):
    tokenizer1 = BertTokenizer.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name)
    config.num_hidden_layers = 2
    config.num_labels = 7
    config.output_attentions = True
    model1 = BertForSequenceClassification.from_pretrained(
        model_name, config=config)

    model1.load_state_dict(torch.load(
        weight_file, map_location=torch.device('cpu')), strict=False)
    model1.eval()
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # 这里的路径传上一节保存的onnx模型地址
    session = InferenceSession(
        onnx_file, sess_options=options, providers=["CPUExecutionProvider"]
    )

    session.disable_fallback()

    sentence = "程序员"
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
            0), attention_mask=attention_mask.unsqueeze(0), token_type_ids=token_type_ids.unsqueeze(0))
        # attention=outputs.attentions
        # tokens=tokenizer1.convert_ids_to_tokens(encoding['input_ids'][0])
        # model_view(attention, tokens)  # Display model view
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


if __name__ == '__main__':
    model_name='./models/bert-roberta'
    weight_file='models/Job.pt'
    onnx_file=export_onnx(model_name,weight_file)
    test(model_name,weight_file,onnx_file)