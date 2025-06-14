# Biomedical Image Analysis / Human Face Recognition

***Topic01_Team01***


The following README contains explanations and instructions for the bioinformatics project *Human Face Recognition* of the summer term 2025.

All in-depth information about the *Data Analysis 2025* module and the given project overview and guidelines can be found here.

- [Information about the Module *Data Analysis 2025*](https://github.com/maiwen-ch/2025_Data_Analysis_Project?tab=readme-ov-file) 
- [Project Overview and Guidelines](https://github.com/maiwen-ch/2025_Data_Analysis_Topic_01_Biomedical_Image_Analysis?tab=readme-ov-file#project-overview-and-guidelines) 


---
## Background 

As image analysis is becoming increasingly important in medicine and biology, our task was to familiarize ourselves with the procedures, methods and theoretical practices and apply them to the Yale Face dataset. 

This project implements a pipeline using:

- **Hold-Out Validation** to split the dataset into training/test subsets.
- **Principal Componenet Analyis (PCA)** to reduce high-dimensional image data into a lower-dimensional subspace (Eigenfaces). 
- **K-Nearest Neighbour (KNN)** to assign each image to the individual it belongs to.

The used *Yale Face Dataset* contains 165 gray scale `.gif` files of uniform size includes multiple individuals with varying facial expressions and lighting conditions.

---
## Git Organization 

**The repository is structured as follows:**
 
[datasets](https://github.com/BiomedicalImageAnalysis2025/topic01_team01/tree/main/datasets) - Contains the raw and unprocessed images files.

[functions](https://github.com/BiomedicalImageAnalysis2025/topic01_team01/tree/main/functions) - Custom implementation of the *Preprocessing-*, *PCA-*, *KNN-* and *Furtheranalysis-* functions.

[main.ipynb](https://github.com/BiomedicalImageAnalysis2025/topic01_team01/blob/main/main.ipynb)- The jupyter notebook explains all the methods including the results of each step with visualizations and graphics.

`README.md` - The current file. Provides an overview of the project.

---
## Instructions 

To review this project and to comprehend each step you need to install following packages to run the code on your device:

- **numpy**
- **os**
- **PIL**

And for plotting and reproduction of the graphics:

- **seaborn**
- **plotly**
- **matplotlib.pyplot**

If all packages were installed you can check out the `main.ipynb`. The Jupyter Notebook file explains all the used methods in detail and visualize the results after each step. Therefore it does not provide any kind of code exept the chunks used for plottin.

For deeper insights how each function was implemented and how the code works, click on [functions](https://github.com/BiomedicalImageAnalysis2025/topic01_team01/tree/main/functions). As every line of code is commented very detailed no uncertainties should be left. If so and you want to learn more about the theory of each method, sources to books and papers were also provided in each section and in the appendix.

BEISPIELQUELLE:

[SVD and PCA](https://databookuw.com/databook.pdf)
etc....

---
