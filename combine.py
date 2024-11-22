import onnx
from onnx import helper

# ONNXファイルのパス
seg_model_path = "onnx/segmodel_from_pt.onnx"
depth_model_path = "onnx/depthmodel_from_pt.onnx"
output_model_path = "onnx/combined_with_mass_model.onnx"

# 1. 各 ONNX モデルを読み込む
seg_model = onnx.load(seg_model_path)
depth_model = onnx.load(depth_model_path)

# 2. それぞれのグラフを取得
seg_graph = seg_model.graph
depth_graph = depth_model.graph

# 3. ノードや初期化子をリストに変換して結合
seg_nodes = list(seg_graph.node)
depth_nodes = list(depth_graph.node)
combined_nodes = []

# Segmodel のノード出力名をリネーム
for node in seg_nodes:
    for i in range(len(node.output)):
        node.output[i] += "_seg"
    combined_nodes.append(node)

# Depthmodel のノード出力名をリネーム
for node in depth_nodes:
    for i in range(len(node.output)):
        node.output[i] += "_depth"
    combined_nodes.append(node)

# 初期化子を結合
combined_initializers = list(seg_graph.initializer) + list(depth_graph.initializer)

# 入力はSegmodelとDepthmodelで共通
combined_inputs = list(seg_graph.input)

# 出力をリネームして取得
seg_outputs = [f"{output.name}_seg" for output in seg_graph.output]
depth_outputs = [f"{output.name}_depth" for output in depth_graph.output]

# MassModel のノードを作成
mass_output_name = "mass_output"
mass_node = helper.make_node(
    "Mul",  # MassModel の計算 (例: 積)
    inputs=seg_outputs + depth_outputs,  # Segmodel と Depthmodel の出力を入力として使用
    outputs=[mass_output_name],
    name="MassModel",
)
combined_nodes.append(mass_node)

# 出力を定義
combined_outputs = [helper.make_tensor_value_info(mass_output_name, onnx.TensorProto.FLOAT, None)]

# 4. 新しいグラフを作成
combined_graph = helper.make_graph(
    nodes=combined_nodes,  # ノードを結合
    name="CombinedModelWithMass",  # 新しいモデル名
    inputs=combined_inputs,  # 入力は共通
    outputs=combined_outputs,  # MassModel の出力
    initializer=combined_initializers,  # 初期化子を結合
)

# 6. 新しいモデルを作成
combined_model = helper.make_model(combined_graph, opset_imports=[helper.make_opsetid("", 21)])  # ONNX opset version)

# 7. モデルを保存
onnx.save(combined_model, output_model_path)
print(f"結合された ONNX モデルを保存しました: {output_model_path}")
