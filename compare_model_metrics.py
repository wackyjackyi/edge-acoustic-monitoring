import os
import re
import pandas as pd
import matplotlib.pyplot as plt


def parse_metrics_txt(path: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    m_acc = re.search(r"Test accuracy:\s*([0-9.]+)", text)
    if not m_acc:
        raise ValueError(f"Cannot find 'Test accuracy' in: {path}")
    acc = float(m_acc.group(1))

    f1_values = []
    for line in text.splitlines():
        line = line.strip()
        if "F1=" in line:
            m = re.search(r"F1=([0-9.]+)", line)
            if m:
                f1_values.append(float(m.group(1)))

    if not f1_values:
        raise ValueError(f"Cannot find per-class F1 values in: {path}")

    macro_f1 = sum(f1_values) / len(f1_values)
    return acc, macro_f1


def main():
    results_dir = os.path.join("data", "results")
    os.makedirs(results_dir, exist_ok=True)

    bc_metrics = os.path.join(results_dir, "metrics.txt")
    mb_metrics = os.path.join(results_dir, "matchboxnet_metrics.txt")

    bc_acc, bc_mf1 = parse_metrics_txt(bc_metrics)
    mb_acc, mb_mf1 = parse_metrics_txt(mb_metrics)

    df = pd.DataFrame([
        {"model": "BC-ResNet", "accuracy": bc_acc, "macro_f1": bc_mf1},
        {"model": "MatchboxNet", "accuracy": mb_acc, "macro_f1": mb_mf1},
    ])

    out_csv = os.path.join(results_dir, "model_comparison.csv")
    out_png = os.path.join(results_dir, "model_comparison.png")

    df.to_csv(out_csv, index=False)

    plt.figure(figsize=(7, 4))
    x = range(len(df))
    plt.bar([i - 0.2 for i in x], df["accuracy"], width=0.4, label="Accuracy")
    plt.bar([i + 0.2 for i in x], df["macro_f1"], width=0.4, label="Macro-F1")
    plt.xticks(list(x), df["model"])
    plt.ylim(0, 1.0)
    plt.title("Model Comparison on ESC-50 (Phase 1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print("Saved:", out_csv)
    print("Saved:", out_png)
    print(df)


if __name__ == "__main__":
    main()