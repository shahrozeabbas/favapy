Using FAVA as a Python library
------------------------------

FAVA accepts either an ``AnnData`` object or a ``pandas.DataFrame`` with genes as rows and cells/samples as columns.

.. code-block:: python

   from favapy import FAVA

   model = FAVA(
       data,
       n_hidden=None,
       n_latents=None,
       log2_normalization=True,
   )

   z_mean = model.cook(max_epochs=50, batch_size=32).get_latent_representation()

   network = model.get_association_network(
       metric='pearson',
       interaction_count=100_000,
       cc_cutoff=None,
   )

For AnnData inputs stored in a non-default layer:

.. code-block:: python

   model = FAVA(adata, layer='counts')

Refer to the tutorials notebook for a full walkthrough.
