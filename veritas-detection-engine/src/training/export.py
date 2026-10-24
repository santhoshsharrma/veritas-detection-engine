import torch


def export_onnx(model, path: str, size: int = 224) -> None:
    model.eval().cpu()
    dummy = torch.zeros(1, 3, size, size)
    torch.onnx.export(
        model, dummy, path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
