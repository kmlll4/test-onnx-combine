import torch
from model import Segmodel, Depthmodel, MassModel

# モデルパス
seg_model_path = "weights/segmodel.pt"
depth_model_path = "weights/depthmodel.pt"
mass_model_path = "weights/massmodel.pt"

# モデルのインスタンスを作成
seg_model = Segmodel()
depth_model = Depthmodel()
mass_model = MassModel()

# state_dict をロード
seg_model.load_state_dict(torch.load(seg_model_path))
depth_model.load_state_dict(torch.load(depth_model_path))
mass_model.load_state_dict(torch.load(mass_model_path))
print("PyTorchモデルをロードしました: segmodel.pt, depthmodel.pt, massmodel.pt")

# ダミー入力 (モデルの入力サイズに合わせる)
dummy_input_1 = torch.randn(1, 3, 4, 4)  # バッチサイズ=1, チャンネル=3, 高さ=4, 幅=4
dummy_input_2 = torch.randn(1, 3, 4, 4)  # MassModel用の2つ目の入力

# ONNXファイルパス
seg_onnx_file = "onnx/segmodel_from_pt.onnx"
depth_onnx_file = "onnx/depthmodel_from_pt.onnx"
mass_onnx_file = "onnx/massmodel_from_pt.onnx"

# Segmodel を ONNX にエクスポート
torch.onnx.export(
    seg_model,  # ロードした PyTorchモデル
    dummy_input_1,  # ダミー入力
    seg_onnx_file,  # 出力ファイルパス
    export_params=True,  # モデルパラメータをエクスポート
    opset_version=11,  # ONNX opset version
    do_constant_folding=True,  # 定数フォールディングの有効化
    input_names=["input"],  # 入力ノード名
    output_names=["output"],  # 出力ノード名
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # 可変長の次元
)
print(f"ONNXモデルをエクスポートしました: {seg_onnx_file}")

# Depthmodel を ONNX にエクスポート
torch.onnx.export(
    depth_model,  # ロードした PyTorchモデル
    dummy_input_1,  # ダミー入力
    depth_onnx_file,  # 出力ファイルパス
    export_params=True,  # モデルパラメータをエクスポート
    opset_version=11,  # ONNX opset version
    do_constant_folding=True,  # 定数フォールディングの有効化
    input_names=["input"],  # 入力ノード名
    output_names=["output"],  # 出力ノード名
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # 可変長の次元
)
print(f"ONNXモデルをエクスポートしました: {depth_onnx_file}")

# MassModel を ONNX にエクスポート
torch.onnx.export(
    mass_model,  # ロードした PyTorchモデル
    (dummy_input_1, dummy_input_2),  # ダミー入力 (複数の入力)
    mass_onnx_file,  # 出力ファイルパス
    export_params=True,  # モデルパラメータをエクスポート
    opset_version=11,  # ONNX opset version
    do_constant_folding=True,  # 定数フォールディングの有効化
    input_names=["input1", "input2"],  # 入力ノード名
    output_names=["output"],  # 出力ノード名
    dynamic_axes={
        "input1": {0: "batch_size"},
        "input2": {0: "batch_size"},
        "output": {0: "batch_size"},
    },  # 可変長の次元
)
print(f"ONNXモデルをエクスポートしました: {mass_onnx_file}")
