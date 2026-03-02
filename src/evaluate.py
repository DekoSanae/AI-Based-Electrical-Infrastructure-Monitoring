from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plots and saves a confusion matrix.
    """

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Mostrar valores dentro de la matriz
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_model(model, test_gen):

    print("Starting evaluation...")

    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes

    print("\nClassification Report:\n")
    report = classification_report(y_true, y_pred, output_dict=True)
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)

    # Crear carpeta models si no existe
    os.makedirs("models", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Guardar métricas en JSON
    results = {
        "test_accuracy": float(report["accuracy"]),
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }

    json_path = f"models/results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    # Guardar imagen de matriz de confusión
    cm_path = f"models/confusion_matrix_{timestamp}.png"
    class_names = list(test_gen.class_indices.keys())
    plot_confusion_matrix(cm, class_names, cm_path)

    print(f"\nResults saved to: {json_path}")
    print(f"Confusion matrix saved to: {cm_path}")

    return report["accuracy"], report, cm
    