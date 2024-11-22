import onnxruntime as ort
import numpy as np

# モデルのパス
combined_model_path = "onnx/combined_with_mass_model.onnx"
seg_model_path = "onnx/segmodel_from_pt.onnx"
depth_model_path = "onnx/depthmodel_from_pt.onnx"

# テスト用のダミー入力 (同じ入力を使用)
dummy_input = np.random.randn(1, 3, 4, 4).astype(np.float32)


# 1. 結合モデルの出力を計算
def run_combined_model(model_path, input_name, dummy_input):
    # ONNX Runtimeで結合モデルをロード
    session = ort.InferenceSession(model_path)
    # 結合モデルの推論
    output = session.run(None, {input_name: dummy_input})
    return output


# 結合モデルの推論
combined_session = ort.InferenceSession(combined_model_path)
combined_input_name = combined_session.get_inputs()[0].name  # 入力名を取得
combined_output = run_combined_model(combined_model_path, combined_input_name, dummy_input)

# MassModel の出力 (結合モデルの出力は mass_output のみ)
mass_output = combined_output[0]

# 2. 単独モデルでの出力を計算
# Segmodel の出力
seg_session = ort.InferenceSession(seg_model_path)
seg_input_name = seg_session.get_inputs()[0].name
seg_output = seg_session.run(None, {seg_input_name: dummy_input})[0]  # 単独モデルの出力

# Depthmodel の出力
depth_session = ort.InferenceSession(depth_model_path)
depth_input_name = depth_session.get_inputs()[0].name
depth_output = depth_session.run(None, {depth_input_name: dummy_input})[0]  # 単独モデルの出力

# MassModel の期待される出力を計算 (Segmodel と Depthmodel の出力を使う)
expected_mass_output = seg_output * depth_output

# 3. 結果を比較
print("MassModelの出力が期待通り:", np.allclose(mass_output, expected_mass_output))

# 各出力の詳細表示（必要に応じて）
print("\nMassModel出力差分:", mass_output - expected_mass_output)
