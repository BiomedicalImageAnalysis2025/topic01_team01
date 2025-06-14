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

The used *Yale Face Dataset* contains 165 gray scaele `.gif` files of uniform size includes multiple individuals with varying facial expressions and lighting conditions.

---
## Git Organization 

**The repository is structured as follows:**
 
[datasets](https://github.com/BiomedicalImageAnalysis2025/topic01_team01/tree/main/datasets) 
- Contains the raw and unprocessed images files.
[functions](https://github.com/BiomedicalImageAnalysis2025/topic01_team01/tree/main/functions)
- Custom implementation of the *Preprocessing-*, *PCA-*, *KNN-* and *Furtheranalysis-* functions.
[main.ipynb](https://github.com/BiomedicalImageAnalysis2025/topic01_team01/blob/main/main.ipynb)
- The jupyter notebook explains all the methods including the results of each step with visualizations and graphics.
`README.md`
- The current file. Provides an overview of the project.

---
## Instructions 
</br>
</br>

---
