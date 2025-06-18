# Biomedical Image Analysis / Human Face Recognition

***Topic01_Team01***


The following README contains explanations and instructions for the bioinformatics project *Human Face Recognition* of the summer term 2025.

All in-depth information about the *Data Analysis 2025* module and the given project overview and guidelines can be found here.

- [Information about the Module *Data Analysis 2025*](https://github.com/maiwen-ch/2025_Data_Analysis_Project?tab=readme-ov-file) 
- [Project Overview and Guidelines](https://github.com/maiwen-ch/2025_Data_Analysis_Topic_01_Biomedical_Image_Analysis?tab=readme-ov-file#project-overview-and-guidelines) 


---
## Background 

As image analysis is becoming increasingly important in medicine and biology, our task was to familiarize ourselves with the procedures, methods and theoretical practices and apply them to the *Yale Face dataset A*. 

This project implements a pipeline using:

- **Hold-Out Validation** to split the dataset into training/test subsets.
- **Principal Componenet Analyis (PCA)** to reduce high-dimensional image data into a lower-dimensional subspace (Eigenfaces). 
- **K-Nearest Neighbour (KNN)** to assign each image to the individual it belongs to.

The used *Yale Face Dataset A* contains 165 gray scale `.gif` files of uniform size includes multiple individuals with varying facial expressions and lighting conditions.

---
## Git Organization 

**The repository is structured as follows:**
 
 nicht vergessen die links zu aktualisieren 


[data](https://github.com/BiomedicalImageAnalysis2025/topic01_team01/tree/main/datasets) - Contains the raw and unprocessed images files.

[functions](https://github.com/BiomedicalImageAnalysis2025/topic01_team01/tree/main/functions) - Custom implementation of the *Preprocessing-*, *PCA-*, *KNN-* and *Evaluation-* functions.

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
- **pandas**
- **matplotlib.pyplot**
- **skitlern.metrics**

If all packages were installed you can check out the `main.ipynb`. The Jupyter Notebook file explains all the used methods in detail and visualize the results after each step. Therefore it does not provide any kind of code exept the chunks used for plotting.

For deeper insights click on [functions](https://github.com/BiomedicalImageAnalysis2025/topic01_team01/tree/main/functions). As every line of code is commented very detailed no uncertainties should be left. If so and you want to learn more about the theory of each method, sources to books and papers were also provided in each section and in the appendix.


---

## Literature 

- Gerbrands, J.J. "On the relationships between SVD, KLT and PCA." Pattern Recognition (1981), vol. 14, issues 1-6, pp 375-381
Belhumeur, P.N., Hespanha, J.P. and Kriegman, D. "Eigenfaces vs. Fisherfaces: Recognition Using Class Specific Linear Projection." IEEE Transactions on Pattern Analysis and Machine Intelligence (1997), vol. 19, pp 711-720.
- Netzer, Y. et al. "Reading Digits in Natural Images with Unsupervised Feature Learning." Proceedings of the Workshop on Neural Information Processing Systems (2011)
- Gareth, J. et al. "An introduction to statistical learning." Springer New York (2013), Chapter 4.4
- Hastie, T., Tibshirani, R. and Friedman, J.H. The Elements of Statistical Learning: Data Mining, Inference, and Prediction. Springer Series in Statistics (2009), 2nd ed., Springer, New York.
- Brunton, S.L. Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control. Cambridge University Press (2019), Cambridge, UK.
- Turk, M., & Pentland, A. (1991). Eigenfaces for recognition. Journal of Cognitive Neuroscience, 3(1), 71–86.