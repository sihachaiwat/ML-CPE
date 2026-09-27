import json
import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

BATCH_SIZES = [16, 32, 64]


def plot_single_history(json_path, save_path, batch_size):
    """อ่านไฟล์ history.json ของแต่ละ Batch Size มาพล็อตกราฟ Accuracy และ Loss"""
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, "r") as f:
        history = json.load(f)

    # ดึงข้อมูล Accuracy และ Loss
    acc = history.get("accuracy", history.get("acc", []))
    val_acc = history.get("val_accuracy", history.get("val_acc", []))
    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])

    epochs = range(len(acc))

    # สร้างรูปขนาด 2 กราฟฝั่งละด้าน (Accuracy และ Loss)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- กราฟ Accuracy ---
    ax1.plot(epochs, acc, label="train", color="#1f77b4", linewidth=1.5)
    ax1.plot(epochs, val_acc, label="validation", color="#ff7f0e", linewidth=1.5)
    ax1.set_title(f"Accuracy (Batch Size {batch_size})", fontsize=12)
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Accuracy", fontsize=10)
    ax1.legend(loc="lower right")

    # --- กราฟ Loss ---
    ax2.plot(epochs, loss, label="train", color="#1f77b4", linewidth=1.5)
    ax2.plot(epochs, val_loss, label="validation", color="#ff7f0e", linewidth=1.5)
    ax2.set_title(f"Loss (Batch Size {batch_size})", fontsize=12)
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Loss", fontsize=10)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    for bs in BATCH_SIZES:
        json_path = os.path.join(OUTPUT_DIR, f"history_bs_{bs}.json")
        save_path = os.path.join(OUTPUT_DIR, f"training_history_bs_{bs}.png")

        plot_single_history(json_path, save_path, bs)


if __name__ == "__main__":
    main()