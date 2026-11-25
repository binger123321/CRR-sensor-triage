"""复现作者 predict.py：用作者 checkpoint 对 578 个预测事件分类。"""
import sys, time, json, numpy as np, torch
sys.path.insert(0, '.')
from safe_io import load_pkl, load_weights
from network.model import SincNetModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
NAMES = ['微震', '爆破', '噪声']
def evaluate(model, data, bs=16, dev='cpu'):
    model.eval(); P = []
    with torch.no_grad():
        for i in range(0, len(data), bs):
            x = torch.tensor(np.array([np.array(e, dtype=np.float32) for e, _ in data[i:i+bs]]))
            P.append(model(x.to(dev)).cpu().numpy())
    P = np.concatenate(P); y = np.array([l for _, l in data]); yp = P.argmax(1)
    p, r, f1, _ = precision_recall_fscore_support(y, yp, average='macro')
    fpr, tpr, _ = roc_curve(label_binarize(y, classes=[0,1,2]).ravel(), P.ravel())
    return dict(n=len(y), accuracy=accuracy_score(y, yp), precision=p, recall=r, f1=f1, auc_micro=auc(fpr, tpr),
                confusion=confusion_matrix(y, yp).tolist(),
                per_class_recall={NAMES[k]: float((yp[y==k]==k).mean()) for k in range(3)}), P
if __name__ == '__main__':
    m = SincNetModel(4); m.load_state_dict(load_weights(sys.argv[1] if len(sys.argv)>1 else 'resources/checkpoint.pth'))
    d = load_pkl('resources/signal_selected_predict_data.pkl')
    t = time.time(); res, P = evaluate(m, d); res['seconds'] = time.time() - t
    print(json.dumps(res, ensure_ascii=False, indent=1))
    import os; os.makedirs('outputs', exist_ok=True); json.dump(res, open('outputs/predict_result.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)
