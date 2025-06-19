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

# missclassified images plot with matplotlib

# Determine misclassified indices
#zipping the true labels and predictions to find misclassified images
misclassified_indices = [i for i, (true, pred) in enumerate(zip(test_labels, predictions)) if true != pred]
# Print the number of misclassified images and their indices
print(f"Total misclassified images found: {len(misclassified_indices)}")
print(misclassified_indices)
# Let's plot up to 16 misclassified images (or fewer if not available)
num_to_plot = min(16, len(misclassified_indices))
# for this a plot with subplots is created
plt.figure(figsize=(12, 12))
for idx, mis_idx in enumerate(misclassified_indices[:num_to_plot]):
    plt.subplot(4, 4, idx+1)
    plt.imshow(original_shape_test_images[mis_idx], cmap='gray') # assuming grayscale images
    plt.title(f"True: {test_labels[mis_idx]}\nPredicted: {predictions[mis_idx]}")
    plt.axis('off')
plt.suptitle("Misclassified Test Images", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


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