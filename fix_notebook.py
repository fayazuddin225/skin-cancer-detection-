import json

notebook_path = r"D:\dataset\HAM10000_split_balanced\val\dataset.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell with the ROC curve plotting code
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('roc_curve(y_test_bin[:, i]' in line for line in source):
            new_source = []
            skip = False
            for line in source:
                if 'y_test_bin = label_binarize' in line:
                    new_source.append(line)
                    continue
                if '# ROC curves' in line:
                    new_source.append(line)
                    new_source.append('    plt.figure()\n')
                    new_source.append('    if num_classes == 2:\n')
                    new_source.append('        fpr, tpr, _ = roc_curve(y_test, svm_probs[:, 1])\n')
                    new_source.append('        auc_val = roc_auc_score(y_test, svm_probs[:, 1])\n')
                    new_source.append('        plt.plot(fpr, tpr, label=f"{class_names[1]} (AUC={auc_val:.3f})")\n')
                    new_source.append('    else:\n')
                    new_source.append('        for i in range(num_classes):\n')
                    new_source.append('            fpr, tpr, _ = roc_curve(y_test_bin[:, i], svm_probs[:, i])\n')
                    new_source.append('            auc_i = roc_auc_score(y_test_bin[:, i], svm_probs[:, i])\n')
                    new_source.append('            plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={auc_i:.3f})")\n')
                    skip = True
                    continue
                if '# PR curves' in line:
                    new_source.append('    plt.plot([0,1],[0,1],"--")\n')
                    new_source.append('    plt.xlabel("False Positive Rate")\n')
                    new_source.append('    plt.ylabel("True Positive Rate")\n')
                    new_source.append('    plt.title("ROC Curves (SVM on CNN Features)")\n')
                    new_source.append('    plt.legend()\n')
                    new_source.append('    plt.show()\n')
                    new_source.append('\n')
                    new_source.append(line)
                    new_source.append('    plt.figure()\n')
                    new_source.append('    if num_classes == 2:\n')
                    new_source.append('        prec, rec, _ = precision_recall_curve(y_test, svm_probs[:, 1])\n')
                    new_source.append('        ap = average_precision_score(y_test, svm_probs[:, 1])\n')
                    new_source.append('        plt.plot(rec, prec, label=f"{class_names[1]} (AP={ap:.3f})")\n')
                    new_source.append('    else:\n')
                    new_source.append('        for i in range(num_classes):\n')
                    new_source.append('            prec, rec, _ = precision_recall_curve(y_test_bin[:, i], svm_probs[:, i])\n')
                    new_source.append('            ap = average_precision_score(y_test_bin[:, i], svm_probs[:, i])\n')
                    new_source.append('            plt.plot(rec, prec, label=f"{class_names[i]} (AP={ap:.3f})")\n')
                    skip = True
                    continue
                if skip:
                    if 'plt.show()' in line or 'plt.legend()' in line or 'plt.title' in line or 'plt.xlabel' in line or 'plt.ylabel' in line or 'plt.plot' in line or 'roc_curve' in line or 'roc_auc_score' in line or 'precision_recall_curve' in line or 'average_precision_score' in line or 'for i in range' in line:
                        if '# PR curves' not in line and 'plt.figure()' not in line:
                           # Check if we are past the plotting block
                           if 'plt.show()' in line and skip:
                               if source.index(line) > source.index([l for l in source if '# PR curves' in l][0]):
                                   skip = False
                        continue
                    else:
                        # Found line after the block
                        skip = False
                        new_source.append(line)
                else:
                    new_source.append(line)
            cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook fixed successfully.")
