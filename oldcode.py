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