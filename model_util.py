import torch
from model import Segmodel, Depthmodel, MassModel


# モデル保存
def save_models(seg_model, depth_model, mass_model, seg_path, depth_path, mass_path):
    """
    モデルを指定したパスに保存する
    """
    torch.save(seg_model.state_dict(), seg_path)
    torch.save(depth_model.state_dict(), depth_path)
    torch.save(mass_model.state_dict(), mass_path)


# モデル読み込み
def load_models(seg_model, depth_model, mass_model, seg_path, depth_path, mass_path):
    """
    指定したパスからモデルのパラメータをロードする
    """
    seg_model.load_state_dict(torch.load(seg_path))
    depth_model.load_state_dict(torch.load(depth_path))
    mass_model.load_state_dict(torch.load(mass_path))


if __name__ == "__main__":
    # モデルの初期化
    seg_model = Segmodel()
    depth_model = Depthmodel()
    mass_model = MassModel()

    # 保存パス
    seg_model_path = "weights/segmodel.pt"
    depth_model_path = "weights/depthmodel.pt"
    mass_model_path = "weights/massmodel.pt"

    # モデルを保存
    save_models(seg_model, depth_model, mass_model, seg_model_path, depth_model_path, mass_model_path)
    print(f"Models saved to {seg_model_path} and {depth_model_path}")

    # モデルのロードテスト
    loaded_seg_model = Segmodel()
    loaded_depth_model = Depthmodel()
    loaded_mass_model = MassModel()
    load_models(
        loaded_seg_model, loaded_depth_model, loaded_mass_model, seg_model_path, depth_model_path, mass_model_path
    )

    # テスト用データ
    test_input = torch.randn(1, 3, 4, 4)

    # オリジナルモデルとロードモデルの結果比較
    original_seg_output = seg_model(test_input)
    loaded_seg_output = loaded_seg_model(test_input)
    assert torch.equal(original_seg_output, loaded_seg_output), "Segmodel outputs do not match!"

    original_depth_output = depth_model(test_input)
    loaded_depth_output = loaded_depth_model(test_input)
    assert torch.equal(original_depth_output, loaded_depth_output), "Depthmodel outputs do not match!"

    original_mass_output = mass_model(test_input, test_input)
    loaded_mass_output = loaded_mass_model(test_input, test_input)
    assert torch.equal(original_mass_output, loaded_mass_output), "Massmodel outputs do not match!"

    print("Models loaded successfully and outputs match!")
