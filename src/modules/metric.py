import numpy as np
from sklearn.metrics import roc_auc_score

LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery', 'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery', 'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery', 'Right Middle Cerebral Artery', 'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery', 'Right Anterior Cerebral Artery', 'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery', 'Basilar Tip', 'Other Posterior Circulation', 
    'Aneurysm Present'
]

def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    scores = {}

    for i, col_name in enumerate(LABEL_COLS):
        try:
            col_true = y_true[:, i]
            col_pred = y_pred[:, i]
            if len(np.unique(col_true)) > 1:
                scores[col_name] = roc_auc_score(col_true, col_pred)
            else:
                scores[col_name] = 0.5
        except Exception as e:
            print(f"Warning: Could not calculate AUC for {col_name}. Setting to 0.5. Error: {e}")
            scores[col_name] = 0.5

    auc_present = scores['Aneurysm Present']    
    location_aucs = [scores[col] for col in LABEL_COLS if col != 'Aneurysm Present']
    mean_location_auc = np.mean(location_aucs)

    final_score = 0.5 * auc_present + 0.5 * mean_location_auc

    return {
        "score": final_score,
        "auc_present": auc_present,
        "mean_location_auc": mean_location_auc,
        **{f"auc_{col.replace(' ', '_').lower()}": score for col, score in scores.items()}
    }