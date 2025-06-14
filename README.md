# Biomedical Image Analysis / Human Face Recognition

***Topic01_Team01***


The following README contains explanations and instructions for the bioinformatics project *Human Face Recognition* of the summer term 2025.

All in-depth information about the *Data Analysis 2025* module and the given project overview and guidelines can be found here.

- [Information about the Module *Data Analysis 2025*](https://github.com/maiwen-ch/2025_Data_Analysis_Project?tab=readme-ov-file) 
- [Project Overview and Guidelines](https://github.com/maiwen-ch/2025_Data_Analysis_Topic_01_Biomedical_Image_Analysis?tab=readme-ov-file#project-overview-and-guidelines) 




---
## Background 
</br>
</br>
As image analysis is becoming increasingly important in medicine and biology, our task was to familiarize ourselves with the procedures, methods and theoretical practices and apply them to the Yale Face dataset. 
</br>
This project implements a pipeline using:
</br> 
</br>
<ul>
<li> **Hold-Out Validation** to split the dataset into training/test subsets.</li>
<li> **Principal Componenet Analyis (PCA)** to reduce high-dimensional image data into a lower-dimensional subspace (Eigenfaces). 
</li>
<li> **K-Nearest Neighbour (KNN)** to assign each image to the individual it belongs to.</li>
</ul> 
</br>
The used *Yale Face Dataset* contains 165 gray scaele `.gif` files of uniform size includes multiple individuals with varying facial expressions and lighting conditions.

---
## Git Organization 
</br>
</br>
**The repository is structured as follows:**
</br> 
[datasets](https://github.com/BiomedicalImageAnalysis2025/topic01_team01/tree/main/datasets) 
</br>
- Contains the raw and unprocessed images files.
</br>
[functions](https://github.com/BiomedicalImageAnalysis2025/topic01_team01/tree/main/functions)
</br>
- Custom implementation of the *Preprocessing-*, *PCA-*, *KNN-* and *Furtheranalysis-* functions.
</br>
- `README.md`/: The current file. Gives an overview of the project.
</br>
[main.ipynb](https://github.com/BiomedicalImageAnalysis2025/topic01_team01/blob/main/main.ipynb)
</br>
- The jupyter notebook explains all the methods including the results of each step with visualization.
---
## Instructions 
</br>
</br>

---
