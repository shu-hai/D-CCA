# D-CCA: A Decomposition-based Canonical Correlation Analysis for High-dimensional Datasets
This python package implements the D-CCA method proposed in [1]. See [example.py](https://github.com/shu-hai/D-CCA/blob/master/example.py) for details, with Python 3.5 or above.


D-CCA conducts the following decomposition:

<img src="https://render.githubusercontent.com/render/math?math=\boldsymbol{Y}_k=\boldsymbol{X}_k %2B \boldsymbol{E}_k=\boldsymbol{C}_k %2B \boldsymbol{D}_k %2B \boldsymbol{E}_k"> for <img src="https://render.githubusercontent.com/render/math?math=k=1,2">

where <img src="https://render.githubusercontent.com/render/math?math=\boldsymbol{C}_1"> and <img src="https://render.githubusercontent.com/render/math?math=\boldsymbol{C}_2"> share the same latent factors, but <img src="https://render.githubusercontent.com/render/math?math=\boldsymbol{D}_1"> and <img src="https://render.githubusercontent.com/render/math?math=\boldsymbol{D}_2"> have uncorrelated latent factors.

Note that <img src="https://render.githubusercontent.com/render/math?math=\boldsymbol{Y}_k"> should be row-mean centered.

Please cite the article [1] for this package, which is available [here](https://www.researchgate.net/publication/329691934_D-CCA_A_Decomposition-based_Canonical_Correlation_Analysis_for_High-Dimensional_Datasets).

[1] Hai Shu, Xiao Wang & Hongtu Zhu (2020) D-CCA: A Decomposition-based Canonical Correlation Analysis for High-dimensional Datasets. Journal of the American Statistical Association, 115(529): 292-306. [DOI: 10.1080/01621459.2018.1543599](https://doi.org/10.1080/01621459.2018.1543599) 

