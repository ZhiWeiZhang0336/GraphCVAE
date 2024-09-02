from setuptools import Command, find_packages, setup

__lib_name__ = "GraphCVAE"
__lib_version__ = "1.0.3"
__description__ = "GraphCVAE: Uncovering Cell Heterogeneity and Therapeutic Target Discovery through Residual and Contrastive Learning"
__url__ = "https://github.com/ZhiWeiZhang0336/GraphCVAE"
__author__ = "Zhiwei Zhang"
__author_email__ = "2023520218@bipt.eud.cn"
__license__ = "MIT"
__keywords__ = ["Spatial Transcriptomics", "Variational Graph Autoencoder",  "Contrastive Learning", "Spatial Clustering","Deep Learning","Gene Expression",]
__requires__ = ["requests",]

with open("README.md", "r", encoding="utf-8") as f:
    __long_description__ = f.read()

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    packages = ["GraphCVAE"],
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True,
    long_description = """GraphCVAE is a comprehensive model that integrates gene expression, spatial coordinates, and tissue morphology to accurately define spatial domains. The model first processes tissue image data to generate a morphological feature matrix, then combines spatial proximity information and gene expression similarity to enhance gene expression data. GraphCVAE employs multi-layer Graph Convolutional Networks (GCN) and a Variational Autoencoder (VAE) to optimize spatial domain representation. GCN captures complex spatial relationships, while VAE transforms high-dimensional data into a low-dimensional latent space. To improve model robustness, GraphCVAE adopts a contrastive learning approach, distinguishing between original and shuffled spatial gene expression data. The model accurately captures spatial patterns and gene expression features by combining reconstruction loss, KL divergence loss, and contrastive loss. The resulting embeddings are used for downstream analysis, including spatial domain identification, batch effect correction, and other spatial transcriptomics-related tasks, revealing tissue spatial structure and deepening understanding of gene expression patterns.""",
    long_description_content_type="text/markdown"
)