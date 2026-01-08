from ultralytics import YOLO


def main():
    model = YOLO("../../shared_models/yolov8n.pt")

    results = model.train(
        data="../config/data.yaml",
        epochs=4,
        batch=24,
        device="cuda",
        workers=1,
        patience=20,
        # name='my_rps_train_v1_fixed',
    )


if __name__ == "__main__":
    main()
