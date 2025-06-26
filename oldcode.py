# Plot for PC vs variance explained in plotly

#zip is used to create a list of hover texts for each point
#it works by combining the number of PCs and the corresponding cumulative variance
hover_texts = [f"PC: {x}<br>Cumulative Variance: {y:.2f}%" if x != 40 else "" for x, y in zip(n_pcs_for_plot, cumulative_variance_norm)]

# Create the main trace for the curve
curve_trace = go.Scatter(
    x=n_pcs_for_plot,
    y=cumulative_variance_norm,
    mode='lines+markers',
    name='Cumulative Variance Curve',
    hoverinfo ='text',
    text=hover_texts
)

# Highlight the point at 40 PCs by adding an extra trace
highlight_trace = go.Scatter(
    x=[n_components],  # x-coordinate at n PCs
    y=[cumulative_variance_norm[n_components-1]],  # corresponding cumulative variance
    mode='markers',
    name=f'Amount of PCs chosen ({n_components})',
    marker=dict(size=10, color='red'),
    hovertemplate='PC: %{x}<br>Cumulative Variance: %{y:.2f}%<extra></extra>'
)
# Create the plot
fig = go.Figure(data=[curve_trace, highlight_trace])

# Update layout for better readability
fig.update_layout(title={
        "text": "Cumulative Variance Explained by Principal Components",
        "y": 0.925,  # Adjust the vertical position (higher value moves it further up)
        "x": 0.5,   # Keep it centered
        },
    xaxis_title="Number of Principal Components",
    yaxis_title="Cumulative Variance Explained (%)",
    width=1000,
    height=600,
    legend=dict(
        orientation="h",  # Makes the legend horizontal
        y=-0.15,           
        x=0.5,            # Centers the legend horizontally
        xanchor="center"  # Anchors the legend to the center
    )
)

fig.show()



# plot for best k evaluation with plotly

#zip is used to create a list of hover texts for each point
#it works by combining the number of PCs and the corresponding cumulative variance
hover_texts = [f"k: {x}<br>KNN-Accuracy: {y:.2f}%" if x != 1 else "" for x, y in zip(k_values, k_accuracy_values)]

# Create the main trace for the curve
curve_trace = go.Scatter(
    x=k_values,
    y=k_accuracy_values,
    mode='lines+markers',
    name='Accuracy Curve',
    hoverinfo ='text',
    text=hover_texts
)

# Highlight the point at best k by adding an extra trace
highlight_trace = go.Scatter(
    x=[best_k],  # x-coordinate at n PCs
    y=[highest_accuracy],  # corresponding cumulative variance
    mode='markers',
    name=f'Best k ({best_k})',
    marker=dict(size=10, color='red'),
    hovertemplate='k: %{x}<br>KNN-Accuracy: %{y:.2f}%<extra></extra>'
)
# Create the plot
fig = go.Figure(data=[curve_trace, highlight_trace])

# Update layout for better readability
fig.update_layout(title={
        "text": "KNN Classification Accuracy based on Number of Neighbors k",
        "y": 0.925,  # Adjust the vertical position (higher value moves it further up)
        "x": 0.5,   # Keep it centered
        },
    xaxis_title="Number of Neighbors (k)",
    yaxis_title="KNN Classification Accuracy (%)",
    width=1000,
    height=600,
    legend=dict(
        orientation="h",  # Makes the legend horizontal
        y=-0.15,           
        x=0.5,            # Centers the legend horizontally
        xanchor="center"  # Anchors the legend to the center
    )
)

fig.show()

#confusion matrix plot with plotly

# Create annotation text
annotations = [[str(cm[i][j]) for j in range(len(classes))] for i in range(len(classes))]

# Compute false annotations (misclassified entries)
false_annotations = np.sum(cm) - np.trace(cm)  # Sum of non-diagonal entries

# Compute missing annotations (count where diagonal != 3)
# expected_value = 3  # Expected value on the diagonal
# missing_annotations = np.sum(cm.diagonal() != expected_value)

# Generate heatmap using Plotly
fig = ff.create_annotated_heatmap(z=cm,
                                  x=classes.tolist(),
                                  y=classes.tolist(), 
                                  annotation_text=annotations, colorscale='Blues')
# Set labels
fig.update_layout(title={
        "text": f"Confusion Matrix for KNN Classification (k = {k})<br>"
        f"Total missclassified Images: {false_annotations}",
        "y": 0.925,  # Adjust the vertical position (higher value moves it further up)
        "x": 0.5,   # Keep it centered
        "xanchor": "center",
        "yanchor": "top"},
    xaxis_title="Predicted Labels", 
    yaxis_title="True Labels",
    autosize=False, 
    width=900,
    height=650,
    xaxis=dict(side="bottom")
)
fig.show()

# Accuracy vs. PC

from functions.preprocessing import preprocessing
from functions.knn import knn_classifier

# How does the number of principal components affect the accuracy of the model?
pc_range = np.arange(1, 30, 1)

accuracies = []

for pc in pc_range:
    final_train, train_labels, final_test, test_labels, test_arr = preprocessing(folder_path, seed, split_tt_ratio, verbose=False)

    projection_matrix, train_reduced, explained_variance_ratio = svd_pca(final_train, pc, verbose=False)

    test_reduced = pca_transform(final_test, projection_matrix, verbose=False)

    predictions = knn_classifier(train_reduced, train_labels, test_reduced, test_labels, k=4, verbose=False)
    accuracy = np.mean(np.array(predictions) == np.array(test_labels))
    accuracies.append(accuracy)


# Plot the accuracies
df = pd.DataFrame({
    'Amount of Principal Components': pc_range,
    'Accuracy (%)': np.array(accuracies) * 100
})

sns.set_theme(style="whitegrid")

plt.figure(figsize=(11, 6))
sns.scatterplot(data=df, x='Amount of Principal Components', y='Accuracy (%)', marker='o')
plt.title("Accuracy of KNN Classifier with Different Amounts of Principal Components")
plt.ylim(0, 100)
plt.tight_layout()

# Confusion matrix for datset b

# Convert them to NumPy arrays if they aren’t already:
true_labels_B = np.array(test_labels_B)
pred_labels_B = np.array(predictions_B)

# Compute the confusion matrix. It will have shape (n_classes, n_classes)
cm_B = confusion_matrix(true_labels_B, pred_labels_B)

#Determine unique class labels for better tick labeling
classes_labels_B = np.unique(np.concatenate((true_labels_B, pred_labels_B)))

# Create an annotation matrix with empty strings where value is 0
annot = np.where(cm_B != 0, cm_B.astype(str), "")

plt.figure(figsize=(13, 8))
heatmap = sns.heatmap(cm_B, cmap="Blues", xticklabels=classes_labels_B, yticklabels=classes_labels_B, annot = annot, fmt="", cbar=False)

# Customize the colorbar annotations
#cbar_B = heatmap.collections[0].colorbar
# Set specific tick positions 
# Labelpad is used to adjust the distance of the label from the colorbar
#cbar_B.set_label("Predictions per Subject", rotation=270, labelpad=30)

#cbar_B.set_ticks(range(0,18))

plt.xlabel('\nPredicted Labels per Subject')
plt.ylabel('True Labels per Subject\n')
plt.title(label=f"\nConfusion Matrix for KNN Classification (k = {k})\n", fontweight='bold')
plt.show()

# extrat metadata from dataset A for color of PC plots

def metadata(data_path):

    exp_cond = []

    for img_file in os.listdir(data_path):
        # Check if the file is a GIF image, if not, it will be skipped.
        if not img_file.endswith(".gif"):
            continue
        # Extract person’s identifier from the filename.
        # I our case, the identifier is the first part of the filename before the dot.
        # [0] splits the string at the dot and takes the first part.
        exp_cond_single = img_file.split(".")[1]
        exp_cond.append(exp_cond_single)

        # Separate lists
    condition_meta = [f for f in exp_cond if "light" in f.lower()]
    rest_meta = [f for f in exp_cond if "light" not in f.lower()]

    # Create DataFrame
    metadata_A = pd.DataFrame({
        "light_related": pd.Series(condition_meta),
        "others": pd.Series(rest_meta)
    })

    return metadata_A

def metadata_sub(data_path):

    subject_id = []

    for img_file in os.listdir(data_path):
        # Check if the file is a GIF image, if not, it will be skipped.
        if not img_file.endswith(".gif"):
            continue
        # Extract person’s identifier from the filename.
        # I our case, the identifier is the first part of the filename before the dot.
        # [0] splits the string at the dot and takes the first part.
        subject_id_sing = img_file.split(".")[0]
        subject_id.append(subject_id_sing)
    
    # Create DataFrame
    metadata_sub = pd.DataFrame({
        "Subject ID": subject_id
    })

    return metadata_sub

#2 PC plots

# Example PCA DataFrame with first few components
df_first_6 = pd.DataFrame(train_reduced[:, :6], columns=["PC1", "PC2", "PC3", "PC4","PC5","PC6"])


combinations_2 = [("PC1", "PC2"),("PC4", "PC6")]

sns.set_theme(style="ticks")

fig, axes = plt.subplots(2, 1, figsize=(6, 8))
#axes = axes.flatten()  # Flatten to make iteration easy

for ax, (x, y) in zip(axes, combinations_2):
    sns.scatterplot(
        data=df_first_6,
        x=x,
        y=y,
        s=60,
        edgecolor="k",
        alpha = 0.6,
        ax=ax
    )
    ax.set_title(f"\n{x} vs {y}\n", fontweight = "bold")

plt.tight_layout(h_pad=2.0)
plt.savefig("PCplots.svg")
plt.show()