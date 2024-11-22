import torch
import torch.nn as nn


class Segmodel(nn.Module):
    """
    セグメンテーションモデル
    単純な足し算操作を行う
    """

    def __init__(self):
        super(Segmodel, self).__init__()

    def forward(self, x):
        # x + x の操作 (単純な足し算)
        return x + x


class Depthmodel(nn.Module):
    """
    深度モデル
    単純な引き算操作を行う
    """

    def __init__(self):
        super(Depthmodel, self).__init__()

    def forward(self, x):
        # x - x の操作 (単純な引き算)
        return x * x


class MassModel(nn.Module):
    """
    入力が2つのモデル
    2つのテンソルを掛け算する
    """

    def __init__(self):
        super(MassModel, self).__init__()

    def forward(self, x1, x2):
        # x1 * x2 の操作 (2つの入力を掛け算)
        return x1 * x2


# テスト用コード
if __name__ == "__main__":
    # 入力テンソルをランダム生成 (バッチサイズ=1, チャンネル=3, 高さ=2, 幅=2)
    input_tensor_1 = torch.randn(1, 3, 2, 2)
    input_tensor_2 = torch.randn(1, 3, 2, 2)
    print("Input Tensor 1:\n", input_tensor_1)
    print("Input Tensor 2:\n", input_tensor_2)

    # Segmodel のテスト
    seg_model = Segmodel()
    seg_output = seg_model(input_tensor_1)
    print("\nSegmodel Output (Addition):\n", seg_output)

    # Depthmodel のテスト
    depth_model = Depthmodel()
    depth_output = depth_model(input_tensor_1)
    print("\nDepthmodel Output (Square):\n", depth_output)

    # MassModel のテスト
    mass_model = MassModel()
    mass_output = mass_model(input_tensor_1, input_tensor_2)
    print("\nMassModel Output (Multiplication):\n", mass_output)
