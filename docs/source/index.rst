Welcome to favapy’s documentation!
==================================

.. toctree::
   :maxdepth: 2

   Introduction
   Installation
   Using_FAVA_as_a_Python_library
   API
   Tutorials

Introduction
-------------

Protein networks are commonly used for understanding the interplay between proteins in the cell as well as for visualizing omics data. Unfortunately, most existing high-quality networks are heavily biased by data availability, in the sense that well-studied proteins have many more interactions than understudied proteins. To create networks that can help elucidate functions for the latter, we must start from data that are not affected by this literature bias, in other words, from omics data such as single cell RNA-seq (scRNA-seq) and proteomics. While networks can be inferred from such data through simple co-expression analysis, this approach does not work well due to high sparseness (many transcripts/proteins are not consistently observed in each cell/sample) and redundancy (many similar cells/samples are analyzed) of such data. We have therefore developed FAVA, Functional Associations using Variational Autoencoders, which deals with both issues by compressing these high-dimensional data into a dense, low-dimensional latent space. Calculating correlations in this latent space results in much improved networks compared to the original representation for large-scale scRNA-seq and proteomics data from the Human Protein Atlas, and from PRIDE, respectively. Additionally, these networks, which given the nature of the input data should be free of literature bias, have much better coverage of understudied proteins than existing networks.

Installation
-------------

You can install FAVA using pip:

.. code-block:: bash

   pip install favapy


Using FAVA as a Python library
------------------------------

.. code-block:: python

   from favapy import FAVA

   model = FAVA(data, n_hidden=None, n_latents=None)
   network = (
       model.cook(max_epochs=50, batch_size=32)
       .get_association_network(metric='pearson', interaction_count=100_000)
   )

Refer to the tutorials for more examples.
