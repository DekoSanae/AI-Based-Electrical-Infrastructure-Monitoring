import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

def show_portfolio(model, generator, save_path="Sample per class.png"):
    
    class_names = list(generator.class_indices.keys())
    n_classes = len(class_names)
    
    shown = {}
    all_preds = []
    all_true = []
    
    generator.reset()
    
    # Recolectar predicciones globales para accuracy
    for images, labels in generator:
        preds = model.predict(images, verbose=0)
        
        all_preds.extend(np.argmax(preds, axis=1))
        all_true.extend(np.argmax(labels, axis=1))
        
        for i in range(len(images)):
            true_class = np.argmax(labels[i])
            
            if true_class not in shown:
                pred_class = np.argmax(preds[i])
                confidence = np.max(preds[i]) * 100
                correct = true_class == pred_class
                
                shown[true_class] = {
                    "image": images[i],
                    "pred": pred_class,
                    "conf": confidence,
                    "correct": correct,
                    "probs": preds[i]
                }
            
            if len(shown) == n_classes:
                break
        
        if len(shown) == n_classes:
            break
    
    accuracy = accuracy_score(all_true, all_preds)
    
    # ===== FIGURA PROFESIONAL =====
    fig = plt.figure(figsize=(4*n_classes, 6))
    
    for idx, class_id in enumerate(sorted(shown.keys())):
        
        data = shown[class_id]
        
        # Imagen
        ax_img = plt.subplot(2, n_classes, idx+1)
        ax_img.imshow(data["image"])
        ax_img.axis("off")
        
        color = "darkgreen" if data["correct"] else "darkred"
        
        ax_img.set_title(
            f"True: {class_names[class_id]}\n"
            f"Pred: {class_names[data['pred']]} ({data['conf']:.1f}%)",
            fontsize=9,
            fontweight="bold",
            color=color
        )
        
        # Barra de probabilidades
        ax_bar = plt.subplot(2, n_classes, idx+1+n_classes)
        """
        ax_bar.bar(class_names, data["probs"])
        ax_bar.set_ylim(0, 1)
        ax_bar.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        """

        x = np.arange(len(class_names))

        ax_bar.bar(x, data["probs"])
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax_bar.set_ylabel("Probability", fontsize=8)
        ax_bar.tick_params(axis='y', labelsize=8)
    
    plt.suptitle(
        f"AI-Based Electrical Infrastructure Monitoring\n"
        f"Overall Test Accuracy: {accuracy*100:.2f}%",
        fontsize=14,
        fontweight="bold"
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    
    print(f"Professional IEEE-style figure saved as: {save_path}")