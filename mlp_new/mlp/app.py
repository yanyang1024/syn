import pandas as pd
import gradio as gr
import numpy as np

from inference import predict
from train import TrainConfig, run


def train_model(
    epochs,
    batch_size,
    lr,
    weight_decay,
    hidden,
    depth,
    expansion,
    dropout,
    head_dropout,
    patience,
    min_delta,
    grad_clip,
    split,
    checkpoint_dir,
):
    config = TrainConfig(
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        weight_decay=float(weight_decay),
        hidden=int(hidden),
        depth=int(depth),
        expansion=int(expansion),
        dropout=float(dropout),
        head_dropout=float(head_dropout),
        patience=int(patience),
        min_delta=float(min_delta),
        grad_clip=None if grad_clip <= 0 else float(grad_clip),
        split=float(split),
        checkpoint_dir=checkpoint_dir,
    )
    summary = run(config)
    history_df = pd.DataFrame(summary.history)
    metrics_text = (
        f"最佳 Epoch: {summary.best_epoch}\n"
        f"验证 MSE: {summary.best_val_mse:.5f}\n"
        f"验证 MAE(原尺度): {summary.best_metrics_raw['mae']:.5f}\n"
        f"验证 RMSE(原尺度): {summary.best_metrics_raw['rmse']:.5f}"
    )
    return metrics_text, history_df


def _table_to_array(table):
    if table is None:
        return None
    arr = np.array(table, dtype="float32")
    if arr.ndim != 2:
        return None
    # 删除全空行
    mask = ~np.isnan(arr).all(axis=1)
    arr = arr[mask]
    if arr.size == 0:
        return None
    return np.nan_to_num(arr, nan=0.0)


def run_inference(table_data, csv_file, checkpoint_dir):
    if csv_file is not None:
        data_source = csv_file.name
    else:
        arr = _table_to_array(table_data)
        if arr is None:
            raise gr.Error("请上传 CSV 或在表格中填写至少一行数据")
        data_source = arr
    preds = predict(data_source, checkpoint_dir=checkpoint_dir, return_numpy=True)
    columns = [f"y_{i:02d}" for i in range(preds.shape[1])]
    return pd.DataFrame(preds, columns=columns)


with gr.Blocks() as demo:
    gr.Markdown("# 多输出回归：训练与推理面板")
    with gr.Tab("训练"):
        gr.Markdown("调整超参数后点击“开始训练”。")
        with gr.Row():
            with gr.Column():
                epochs = gr.Slider(100, 1000, value=400, step=50, label="Epochs")
                batch_size = gr.Slider(16, 256, value=128, step=16, label="Batch Size")
                lr = gr.Slider(1e-5, 1e-2, value=3e-4, step=1e-5, label="Learning Rate")
                weight_decay = gr.Slider(0.0, 1e-2, value=1e-4, step=1e-5, label="Weight Decay")
                patience = gr.Slider(10, 120, value=60, step=5, label="Early Stopping Patience")
                min_delta = gr.Slider(0.0, 1e-3, value=1e-4, step=1e-5, label="Early Stopping Δ")
                split = gr.Slider(0.6, 0.95, value=0.85, step=0.01, label="Train Split")
                grad_clip = gr.Slider(0.0, 5.0, value=1.0, step=0.1, label="Grad Clip (0 表示关闭)")
            with gr.Column():
                hidden = gr.Slider(64, 512, value=256, step=32, label="Hidden Width")
                depth = gr.Slider(2, 12, value=6, step=1, label="Residual Blocks")
                expansion = gr.Slider(1, 4, value=2, step=1, label="Expansion Ratio")
                dropout = gr.Slider(0.0, 0.5, value=0.2, step=0.05, label="Block Dropout")
                head_dropout = gr.Slider(0.0, 0.5, value=0.1, step=0.05, label="Head Dropout")
                checkpoint_dir = gr.Textbox(value="checkpoints", label="Checkpoint 目录")
        train_btn = gr.Button("开始训练")
        metrics_box = gr.Textbox(label="训练结果", interactive=False, lines=4)
        history_table = gr.Dataframe(label="训练日志", interactive=False)
        train_btn.click(
            train_model,
            inputs=[
                epochs,
                batch_size,
                lr,
                weight_decay,
                hidden,
                depth,
                expansion,
                dropout,
                head_dropout,
                patience,
                min_delta,
                grad_clip,
                split,
                checkpoint_dir,
            ],
            outputs=[metrics_box, history_table],
        )

    with gr.Tab("推理"):
        gr.Markdown("上传 CSV 文件或在表格中填入样本，每行 8 个特征。")
        input_table = gr.Dataframe(
            headers=[f"x_{i}" for i in range(8)],
            row_count=2,
            col_count=8,
            label="手动输入",
        )
        csv_file = gr.File(label="或上传 CSV", file_types=[".csv"])
        ckpt_dir_pred = gr.Textbox(value="checkpoints", label="Checkpoint 目录")
        infer_btn = gr.Button("运行预测")
        output_table = gr.Dataframe(label="预测结果")
        infer_btn.click(run_inference, inputs=[input_table, csv_file, ckpt_dir_pred], outputs=output_table)


if __name__ == "__main__":
    demo.launch()
