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
- **K-Nearest Neighbour (K<sub>NN</sub>)** to assign each image to the individual it belongs to.

The used *Yale Face Dataset A* contains 165 gray scale `.gif` files of uniform size includes multiple individuals with varying facial expressions and lighting conditions.

---
## Git Organization 

**The repository is structured as follows:**
 

[dataA](https://github.com/BiomedicalImageAnalysis2025/topic01_team01/tree/main/dataA) - Contains the raw and unprocessed images files of the original *Yale Face Dataset A*.

[dataB](https://github.com/BiomedicalImageAnalysis2025/topic01_team01/tree/main/dataB) - Contains the raw and unprocessed images files of the *Yale Face Dataset B* for further analysis.

[functions](https://github.com/BiomedicalImageAnalysis2025/topic01_team01/tree/main/functions) - Custom implementation of the *Preprocessing-*, *PCA-*, *KNN-*, *Evaluation-* and *Furtheranalyis* functions.

[main.ipynb](https://github.com/BiomedicalImageAnalysis2025/topic01_team01/blob/main/main.ipynb)- The jupyter notebook explains all the methods including the results of each step with visualizations and graphics.

`README.md` - The current file. Provides an overview of the project.

---
## Instructions 

To review this project and to comprehend each step you need to install following packages to run the code on your device:

- `numpy`
- `os`
- `PIL`
- `scipy.io`

And for plotting and reproduction of the graphics:

- `seaborn`
- `pandas`
- `matplotlib.pyplot`
- `sklearn`
    - `.metrics`
    - `.neighbours`
    - `.model_selection`

If all packages were installed you can check out the `main.ipynb`. The Jupyter Notebook file explains all the used methods in detail and visualize the results after each step. Therefore it does not provide any kind of code exept the chunks used for plotting.

For deeper insights click on [functions](https://github.com/BiomedicalImageAnalysis2025/topic01_team01/tree/main/functions). As every line of code is commented very detailed no uncertainties should be left. If so and you want to learn more about the theory of each method, sources to books and papers were also provided in each section and in the appendix.


---
## Usage of AI and Workload Splitting

AI was used throughout the project as support for validating results, debugging or setting further incentives. However, **no** entire sections were taken over and each line was commented and retraced independently. <br> 
Tools used were:

- `ChatGPT`
- `Github Copilot`
- `Copilot`

The aim of the group was for everyone to implement each function independently so that they could be put together later and any errors made by a member could be corrected. Subsequent tasks, such as creating the plots and the further analysis, were divided between Fedor and Mats, while Kimia and Nicolas were already working on the scientific poster.

---

## Literature 

- Belhumeur, P.N., Hespanha, J.P., & Kriegman, D.J. (1997). Eigenfaces vs. Fisherfaces: recognition using class specific linear projection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 19(7), 711–720. https://doi.org/10.1109/34.598228
- Brunton, S.L. (2019). Data-Driven Science and Engineering. Cambridge University Press.
- Gareth, J., Witten, D., Hastie, T., & Tibshirani, R. (2013). An introduction to statistical learning. Springer, New York. Kapitel 4.4.
- Gerbrands, J.J. (1981). On the relationships between SVD, KLT and PCA. Pattern Recognition, 14(1-6), 375–381.
- Hastie, T., Tibshirani, R., & Friedman, J.H. (2009). The Elements of Statistical Learning. 2nd ed., Springer, New York.
- Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B., & Ng, A.Y. (2011). Reading digits in natural images with unsupervised feature learning. Proceedings of the Workshop on Neural Information Processing Systems.
- Turk, M., & Pentland, A. (1991). Eigenfaces for recognition. Journal of Cognitive Neuroscience, 3(1), 71–86.